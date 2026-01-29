import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import pytz
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer (IST)", layout="wide")
st.title("🚂 BOXN Rake Demurrage Optimization Dashboard (IST)")
st.markdown("""
**New Integration:**
- 📊 **Google Sheet Sync:** Automatically fetching 'Actual' unloading data for today.
- 🔄 **Hybrid Simulation:** Uses Google Sheet data to set Tippler availability, then schedules uploaded CSV rakes.
- 🎯 **Deduplication:** Automatically ignores rakes in the CSV if they are already in the Google Sheet.
""")

# Define IST Timezone
IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def to_ist(dt):
    """Ensures a datetime object is in IST."""
    if pd.isnull(dt): return pd.NaT
    if dt.tzinfo is None:
        return IST.localize(dt)
    return dt.astimezone(IST)

def parse_wagons(val):
    try:
        if '+' in str(val):
            parts = str(val).split('+')
            return sum(int(p) for p in parts if p.strip().isdigit())
        return int(float(val))
    except: return 0

def format_dt(dt):
    """Formats datetime as dd-HH:MM string."""
    if pd.isnull(dt): return ""
    if dt.tzinfo is None: dt = IST.localize(dt)
    return dt.strftime('%d-%H:%M')

def restore_dt(dt_str, ref_dt):
    """Restores datetime from 'dd-HH:MM' string using reference year/month."""
    if not isinstance(dt_str, str) or dt_str.strip() == "": return pd.NaT
    try:
        parts = dt_str.split('-') 
        day = int(parts[0])
        time_parts = parts[1].split(':')
        hour, minute = int(time_parts[0]), int(time_parts[1])
        
        if ref_dt.tzinfo is None: ref_dt = IST.localize(ref_dt)
        
        new_dt = ref_dt.replace(day=day, hour=hour, minute=minute, second=0)
        
        if day < ref_dt.day - 15: 
            new_dt = new_dt + pd.DateOffset(months=1)
        elif day > ref_dt.day + 15:
            new_dt = new_dt - pd.DateOffset(months=1)
            
        return new_dt
    except: return pd.NaT

def format_duration_hhmm(delta):
    if pd.isnull(delta): return ""
    total_seconds = int(delta.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{sign}{hours:02d}:{minutes:02d}"

def check_downtime_impact(tippler_id, proposed_start, downtime_list):
    if proposed_start.tzinfo is None:
        proposed_start = IST.localize(proposed_start)
        
    relevant_dts = [d for d in downtime_list if d['Tippler'] == tippler_id]
    relevant_dts.sort(key=lambda x: x['Start'])
    
    current_start = proposed_start
    changed = True
    while changed:
        changed = False
        for dt in relevant_dts:
            dt_start = dt['Start']
            dt_end = dt['End']
            # If start time falls inside downtime, push it
            if dt_start <= current_start < dt_end:
                current_start = dt_end
                changed = True
    return current_start

# ==========================================
# 3. GOOGLE SHEET ACTUALS LOGIC
# ==========================================

def fetch_google_sheet_actuals(url):
    """
    Reads published Google Sheet CSV.
    Mappings:
    Col E (idx 4): Receipt (Arrival)
    Col F (idx 5): Placement (Start)
    Col G (idx 6): Unload End
    Col H (idx 7): Release
    Col I (idx 8): Unload Duration
    Col O, P, Q, R (idx 14, 15, 16, 17): Tipplers Used (WT1, WT2, WT3, WT4)
    """
    try:
        df_gs = pd.read_csv(url)
        
        # Basic Validation: Ensure enough columns exist
        if len(df_gs.columns) < 18:
            st.error("Google Sheet does not have enough columns (Needs at least up to Column R).")
            return pd.DataFrame(), {}

        # 1. Parse Dates and Filter for TODAY
        # Assuming Rake Name is in Column B (Index 1) or C (Index 2). 
        # Let's try to find a column with "RAKE" or default to index 1.
        rake_col_idx = 1
        for i, col in enumerate(df_gs.columns):
            if "RAKE" in str(col).upper():
                rake_col_idx = i
                break
        
        actuals = []
        today_date = datetime.now(IST).date()

        # Iterate rows
        for _, row in df_gs.iterrows():
            # Parse Arrival (Col E / Index 4)
            arrival_raw = row.iloc[4]
            if pd.isnull(arrival_raw): continue
            
            try:
                arrival_dt = pd.to_datetime(arrival_raw, dayfirst=True) # Assuming DD/MM/YYYY
                if pd.isnull(arrival_dt): continue
                arrival_dt = to_ist(arrival_dt)
            except: continue

            # Filter: ONLY TODAY'S DATA
            if arrival_dt.date() != today_date:
                continue

            rake_name = str(row.iloc[rake_col_idx])
            
            # Times
            start_dt = to_ist(pd.to_datetime(row.iloc[5], dayfirst=True)) # Col F
            end_dt = to_ist(pd.to_datetime(row.iloc[6], dayfirst=True))   # Col G
            release_dt = to_ist(pd.to_datetime(row.iloc[7], dayfirst=True)) # Col H
            
            # Durations
            # User Rule: Total Duration = Release (H) - Receipt (E)
            if pd.notnull(release_dt) and pd.notnull(arrival_dt):
                total_dur = release_dt - arrival_dt
            else:
                total_dur = timedelta(0)

            # Tipplers Used (O, P, Q, R -> Indices 14, 15, 16, 17)
            # Check if cell is not empty/null
            used = []
            if pd.notnull(row.iloc[14]) and str(row.iloc[14]).strip() != "": used.append("T1")
            if pd.notnull(row.iloc[15]) and str(row.iloc[15]).strip() != "": used.append("T2")
            if pd.notnull(row.iloc[16]) and str(row.iloc[16]).strip() != "": used.append("T3")
            if pd.notnull(row.iloc[17]) and str(row.iloc[17]).strip() != "": used.append("T4")
            
            used_str = ", ".join(used)
            
            entry = {
                'Rake': rake_name,
                'Load Type': 'BOXN (Actual)', # Placeholder
                'Wagons': 58, # Default placeholder
                'Status': 'ACTUAL',
                '_Arrival_DT': arrival_dt,
                '_Shunt_Ready_DT': start_dt, # Approx
                '_Form_Mins': 0,
                'Optimization Type': 'Actual (G-Sheet)',
                'Extra Shunt (Mins)': 0,
                'Line Allotted': 'N/A',
                'Line Entry Time': format_dt(arrival_dt),
                'Shunting Complete': format_dt(start_dt),
                'Tippler Start Time': format_dt(start_dt),
                'Finish Unload': format_dt(end_dt),
                'Tipplers Used': used_str,
                'Wait (Tippler)': format_duration_hhmm(start_dt - arrival_dt) if pd.notnull(start_dt) else "",
                'Total Duration': format_duration_hhmm(total_dur),
                'Demurrage': format_duration_hhmm(max(timedelta(0), total_dur - timedelta(hours=7))), # Assuming 7h free time
                # Internal use for state tracking
                '_raw_end_dt': end_dt,
                '_raw_tipplers': used
            }
            
            # Add T1-T4 specific columns for visualization consistency
            for t in ['T1', 'T2', 'T3', 'T4']:
                entry[f"{t} Start"] = format_dt(start_dt) if t in used else ""
                entry[f"{t} End"] = format_dt(end_dt) if t in used else ""
                entry[f"{t} Idle"] = "" # Actuals don't calc idle
            
            actuals.append(entry)

        df_actuals = pd.DataFrame(actuals)
        
        # Calculate Initial State for Simulation based on Actuals
        tippler_init_state = {}
        for t in ['T1', 'T2', 'T3', 'T4']:
            # Find max end time for this tippler in actuals
            t_max = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
            if not df_actuals.empty:
                for _, row in df_actuals.iterrows():
                    if t in row['_raw_tipplers'] and pd.notnull(row['_raw_end_dt']):
                        if row['_raw_end_dt'] > t_max:
                            t_max = row['_raw_end_dt']
            tippler_init_state[t] = t_max

        return df_actuals, tippler_init_state

    except Exception as e:
        st.error(f"Error parsing Google Sheet: {e}")
        return pd.DataFrame(), {}

# ==========================================
# 4. CORE SIMULATION LOGIC
# ==========================================

def calculate_generic_finish(wagons, target_tipplers, ready_time, tippler_state, downtime_list, 
                           rates, wagons_first_batch, inter_tippler_delay):
    if wagons == 0: 
        return ready_time, "", ready_time, {}, timedelta(0)
    
    resources = {t: rates[t] for t in target_tipplers}
    sorted_tipplers = sorted(resources.keys(), key=lambda x: tippler_state[x])
    
    t_primary = sorted_tipplers[0]
    t_secondary = sorted_tipplers[1] if len(sorted_tipplers) > 1 else None
    
    detailed_timings = {}
    used_tipplers_list = []
    
    w_first = min(wagons, wagons_first_batch)
    w_second = wagons - w_first
    
    # Primary Logic
    free_prim = tippler_state[t_primary]
    if free_prim.tzinfo is None: free_prim = IST.localize(free_prim)
    
    proposed_start_prim = max(ready_time, free_prim)
    actual_start_prim = check_downtime_impact(t_primary, proposed_start_prim, downtime_list)
    
    idle_gap_primary = max(timedelta(0), actual_start_prim - free_prim)
    detailed_timings[f"{t_primary}_Idle"] = idle_gap_primary
    
    finish_primary = actual_start_prim + timedelta(hours=(w_first / resources[t_primary]))
    detailed_timings[f"{t_primary}_Start"] = actual_start_prim
    detailed_timings[f"{t_primary}_End"] = finish_primary
    
    overall_finish = finish_primary
    used_tipplers_list.append(t_primary)
    total_idle_cost = idle_gap_primary
    
    # Secondary Logic
    if w_second > 0:
        if t_secondary:
            start_sec_theory = ready_time + timedelta(minutes=inter_tippler_delay)
            free_sec = tippler_state[t_secondary]
            if free_sec.tzinfo is None: free_sec = IST.localize(free_sec)
            
            proposed_start_sec = max(start_sec_theory, free_sec)
            actual_start_sec = check_downtime_impact(t_secondary, proposed_start_sec, downtime_list)
            
            idle_gap_secondary = max(timedelta(0), actual_start_sec - free_sec)
            detailed_timings[f"{t_secondary}_Idle"] = idle_gap_secondary
            
            finish_secondary = actual_start_sec + timedelta(hours=(w_second / resources[t_secondary]))
            
            detailed_timings[f"{t_secondary}_Start"] = actual_start_sec
            detailed_timings[f"{t_secondary}_End"] = finish_secondary
            
            overall_finish = max(finish_primary, finish_secondary)
            used_tipplers_list.append(t_secondary)
            total_idle_cost += idle_gap_secondary
        else:
            finish_extended = finish_primary + timedelta(hours=(w_second / resources[t_primary]))
            detailed_timings[f"{t_primary}_End"] = finish_extended
            overall_finish = finish_extended

    used_str = ", ".join(sorted(used_tipplers_list))
    return overall_finish, used_str, actual_start_prim, detailed_timings, total_idle_cost

def get_line_entry_time(group, arrival, line_groups):
    grp = line_groups[group]
    active = sorted([t for t in grp['line_free_times'] if t > arrival])
    if len(active) < grp['capacity']: return arrival
    return active[len(active) - grp['capacity']]

def find_specific_line(group, entry, status, last_idx):
    cands = [8, 9, 10] if group == 'Group_Lines_8_10' else [11]
    if group == 'Group_Lines_8_10':
        start = (last_idx + 1) % 3
        cands = cands[start:] + cands[:start]
    for l in cands:
        if status[l] <= entry: return l
    return cands[0]

def run_full_simulation_initial(df, params, actuals_df, actuals_state):
    # Unpack params
    rates = {'T1': params['rt1'], 'T2': params['rt2'], 'T3': params['rt3'], 'T4': params['rt4']}
    s_a, s_b = params['sa'], params['sb']
    f_a, f_b = params['fa'], params['fb']
    w_batch, w_delay = params['wb'], params['wd']
    ft_hours = params['ft']
    extra_shunt_cross = params['extra_shunt']
    downtimes = params['downtimes']

    df = df.copy()
    
    # 1. FILTER BOXN ONLY
    if 'LOAD TYPE' in df.columns:
        df = df[df['LOAD TYPE'].astype(str).str.strip().str.upper().str.startswith('BOXN')]
    
    df['wagon_count'] = df['TOTL UNTS'].apply(parse_wagons)
    df['exp_arrival_dt'] = pd.to_datetime(df['EXPD ARVLTIME'], errors='coerce').apply(to_ist)
    
    if 'STTS CODE' in df.columns and 'STTS TIME' in df.columns:
        df['stts_time_dt'] = pd.to_datetime(df['STTS TIME'], errors='coerce').apply(to_ist)
        df['arrival_dt'] = df.apply(lambda r: r['stts_time_dt'] if str(r.get('STTS CODE')).strip()=='PL' and pd.notnull(r['stts_time_dt']) else r['exp_arrival_dt'], axis=1)
    else:
        df['arrival_dt'] = df['exp_arrival_dt']

    df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt').reset_index(drop=True)

    # 2. DEDUPLICATE: Remove rakes already in Actuals
    if not actuals_df.empty:
        actual_rakes = actuals_df['Rake'].astype(str).tolist()
        df = df[~df['RAKE NAME'].astype(str).isin(actual_rakes)]

    if df.empty and actuals_df.empty:
        return pd.DataFrame(), None

    # 3. INITIALIZE STATE
    first_arrival = df['arrival_dt'].min() if not df.empty else datetime.now(IST)
    sim_start_time = first_arrival.replace(hour=0, minute=0, second=0, microsecond=0)
    if sim_start_time.tzinfo is None: sim_start_time = IST.localize(sim_start_time)

    # Use state from Google Sheet Actuals if available, else default to start of day
    tippler_state = actuals_state if actuals_state else {k: sim_start_time for k in ['T1', 'T2', 'T3', 'T4']}
    
    spec_line_status = {8: sim_start_time, 9: sim_start_time, 10: sim_start_time, 11: sim_start_time}
    line_groups = {
        'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}, 
        'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}
    }
    rr_tracker_A = -1 
    assignments = []
    
    # 4. RUN SIMULATION ON REMAINING RAKES
    for _, rake in df.iterrows():
        options = []

        # --- OPTION 1: NATURAL A (Lines 8-10 -> T1/T2) ---
        entry_A = get_line_entry_time('Group_Lines_8_10', rake['arrival_dt'], line_groups)
        ready_A = entry_A + timedelta(minutes=s_a)
        fin_A, used_A, start_A, tim_A, idle_A = calculate_generic_finish(
            rake['wagon_count'], ['T1', 'T2'], ready_A, tippler_state, downtimes, rates, w_batch, w_delay
        )
        dem_A = max(timedelta(0), (fin_A - rake['arrival_dt']) + timedelta(minutes=f_a) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Nat_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_A, 'start': start_A, 
            'fin': fin_A, 'used': used_A, 'timings': tim_A, 'idle': idle_A, 'dem': dem_A, 
            'extra_shunt': 0.0, 'form_mins': f_a, 'type': 'Standard'
        })

        # --- OPTION 2: NATURAL B (Line 11 -> T3/T4) ---
        entry_B = get_line_entry_time('Group_Line_11', rake['arrival_dt'], line_groups)
        ready_B = entry_B + timedelta(minutes=s_b)
        fin_B, used_B, start_B, tim_B, idle_B = calculate_generic_finish(
            rake['wagon_count'], ['T3', 'T4'], ready_B, tippler_state, downtimes, rates, w_batch, w_delay
        )
        dem_B = max(timedelta(0), (fin_B - rake['arrival_dt']) + timedelta(minutes=f_b) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Nat_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_B, 'start': start_B, 
            'fin': fin_B, 'used': used_B, 'timings': tim_B, 'idle': idle_B, 'dem': dem_B, 
            'extra_shunt': 0.0, 'form_mins': f_b, 'type': 'Standard'
        })

        # --- OPTION 3: CROSS A (Lines 8-10 -> T3/T4) ---
        ready_A_Cross = ready_A + timedelta(minutes=extra_shunt_cross)
        fin_AX, used_AX, start_AX, tim_AX, idle_AX = calculate_generic_finish(
            rake['wagon_count'], ['T3', 'T4'], ready_A_Cross, tippler_state, downtimes, rates, w_batch, w_delay
        )
        dem_AX = max(timedelta(0), (fin_AX - rake['arrival_dt']) + timedelta(minutes=f_a) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Cross_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_A_Cross, 'start': start_AX, 
            'fin': fin_AX, 'used': used_AX, 'timings': tim_AX, 'idle': idle_AX, 'dem': dem_AX, 
            'extra_shunt': float(extra_shunt_cross), 'form_mins': f_a, 'type': 'Cross-Transfer'
        })

        # --- OPTION 4: CROSS B (Line 11 -> T1/T2) ---
        ready_B_Cross = ready_B + timedelta(minutes=extra_shunt_cross)
        fin_BX, used_BX, start_BX, tim_BX, idle_BX = calculate_generic_finish(
            rake['wagon_count'], ['T1', 'T2'], ready_B_Cross, tippler_state, downtimes, rates, w_batch, w_delay
        )
        dem_BX = max(timedelta(0), (fin_BX - rake['arrival_dt']) + timedelta(minutes=f_b) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Cross_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_B_Cross, 'start': start_BX, 
            'fin': fin_BX, 'used': used_BX, 'timings': tim_BX, 'idle': idle_BX, 'dem': dem_BX, 
            'extra_shunt': float(extra_shunt_cross), 'form_mins': f_b, 'type': 'Cross-Transfer'
        })

        best_opt = sorted(options, key=lambda x: (x['dem'], x['fin'], x['idle']))[0]

        for k, v in best_opt['timings'].items():
            if 'End' in k:
                tippler_key = k.split('_')[0]
                tippler_state[tippler_key] = v
        
        if best_opt['grp'] == 'Group_Lines_8_10':
            sel_line = find_specific_line(best_opt['grp'], best_opt['entry'], spec_line_status, rr_tracker_A)
            if sel_line in [8,9,10]: rr_tracker_A = {8:0,9:1,10:2}[sel_line]
        else:
            sel_line = 11
        
        clr_mins = line_groups[best_opt['grp']]['clearance_mins']
        line_free = best_opt['entry'] + timedelta(minutes=clr_mins)
        line_groups[best_opt['grp']]['line_free_times'].append(line_free)
        spec_line_status[sel_line] = line_free
        
        wait_delta = best_opt['start'] - best_opt['ready']
        form_delta = timedelta(minutes=best_opt['form_mins'])
        tot_dur = (best_opt['fin'] - rake['arrival_dt']) + form_delta

        row_data = {
            'Rake': rake['RAKE NAME'],
            'Load Type': rake['LOAD TYPE'],
            'Wagons': rake['wagon_count'],
            'Status': rake.get('STTS CODE', 'N/A'),
            '_Arrival_DT': rake['arrival_dt'],
            '_Shunt_Ready_DT': best_opt['ready'],
            '_Form_Mins': best_opt['form_mins'],
            'Optimization Type': best_opt['type'],
            'Extra Shunt (Mins)': best_opt['extra_shunt'],
            'Line Allotted': sel_line,
            'Line Entry Time': format_dt(best_opt['entry']),
            'Shunting Complete': format_dt(best_opt['ready']),
            'Tippler Start Time': format_dt(best_opt['start']),
            'Finish Unload': format_dt(best_opt['fin']),
            'Tipplers Used': best_opt['used'],
            'Wait (Tippler)': format_duration_hhmm(wait_delta),
            'Total Duration': format_duration_hhmm(tot_dur),
            'Demurrage': format_duration_hhmm(best_opt['dem'])
        }

        for t in ['T1', 'T2', 'T3', 'T4']:
             row_data[f"{t} Start"] = format_dt(best_opt['timings'].get(f"{t}_Start", pd.NaT))
             row_data[f"{t} End"] = format_dt(best_opt['timings'].get(f"{t}_End", pd.NaT))
             row_data[f"{t} Idle"] = format_duration_hhmm(best_opt['timings'].get(f"{t}_Idle", pd.NaT))
        
        assignments.append(row_data)

    df_simulated = pd.DataFrame(assignments)
    
    # 5. MERGE ACTUALS + SIMULATED
    if not actuals_df.empty:
        # Drop internal columns from actuals before merge
        actuals_clean = actuals_df.drop(columns=['_raw_end_dt', '_raw_tipplers'], errors='ignore')
        if df_simulated.empty:
            final_df = actuals_clean
        else:
            final_df = pd.concat([actuals_clean, df_simulated], ignore_index=True)
    else:
        final_df = df_simulated

    return final_df, sim_start_time

def recalculate_cascade_reactive(edited_df, free_time_hours, sim_start_dt):
    recalc_rows = []
    
    if sim_start_dt.tzinfo is None:
        sim_start_dt = IST.localize(sim_start_dt)
        
    tippler_state = {k: sim_start_dt for k in ['T1', 'T2', 'T3', 'T4']}
    daily_demurrage_tracker = {}
    
    for _, row in edited_df.iterrows():
        # IF ACTUAL, SKIP RECALC but UPDATE STATE
        if row.get('Status') == 'ACTUAL':
            # Update state so subsequent rows know when tipplers are free
            for t in ['T1', 'T2', 'T3', 'T4']:
                end_str = row.get(f"{t} End")
                if pd.notnull(end_str) and end_str != "":
                    end_dt = restore_dt(end_str, sim_start_dt)
                    if pd.notnull(end_dt) and end_dt > tippler_state[t]:
                        tippler_state[t] = end_dt
            
            # Recalculate Demurrage for Actuals just in case params changed
            arrival = pd.to_datetime(row['_Arrival_DT'])
            if arrival.tzinfo is None: arrival = IST.localize(arrival)
            
            # Parse Duration
            dur_str = row['Total Duration']
            # Reconstruct duration from string hh:mm
            try:
                sign = -1 if dur_str.startswith('-') else 1
                parts = dur_str.replace('-','').split(':')
                dur_td = timedelta(hours=int(parts[0]), minutes=int(parts[1])) * sign
                demurrage = max(timedelta(0), dur_td - timedelta(hours=free_time_hours))
                row['Demurrage'] = format_duration_hhmm(demurrage)
            except: pass
            
            recalc_rows.append(row)
            continue
            
        # --- SIMULATED ROW LOGIC ---
        arrival = pd.to_datetime(row['_Arrival_DT'])
        if arrival.tzinfo is None: arrival = IST.localize(arrival)
        
        ready = pd.to_datetime(row['_Shunt_Ready_DT'])
        if ready.tzinfo is None: ready = IST.localize(ready)
        
        form_mins = float(row['_Form_Mins'])
        
        used_str = str(row['Tipplers Used'])
        current_tipplers = []
        for t in ['T1', 'T2', 'T3', 'T4']:
            if t in used_str: current_tipplers.append(t)
        
        resource_free_at = sim_start_dt
        for t in current_tipplers:
            resource_free_at = max(resource_free_at, tippler_state[t])
        
        valid_start = max(ready, resource_free_at)
        user_start_input = restore_dt(row['Tippler Start Time'], ready)
        
        if pd.notnull(user_start_input):
            final_start = max(valid_start, user_start_input)
        else:
            final_start = valid_start
            
        max_finish = final_start 
        
        for t_id in ['T1', 'T2', 'T3', 'T4']:
            col_start = f"{t_id} Start"
            col_end = f"{t_id} End"
            col_idle = f"{t_id} Idle"
            
            if t_id in current_tipplers:
                prev_finish = tippler_state[t_id]
                gap = final_start - prev_finish
                current_idle = max(timedelta(0), gap)
                row[col_idle] = format_duration_hhmm(current_idle)

                t_end_val = restore_dt(row[col_end], final_start)
                t_start_val_old = restore_dt(row[col_start], final_start)
                
                if pd.notnull(t_end_val) and pd.notnull(t_start_val_old):
                      duration = t_end_val - t_start_val_old
                      if duration < timedelta(0): duration = timedelta(hours=2) 
                else:
                      duration = timedelta(hours=2) 
                
                new_t_end = final_start + duration
                if pd.notnull(t_end_val) and t_end_val > new_t_end:
                    new_t_end = t_end_val

                tippler_state[t_id] = new_t_end
                max_finish = max(max_finish, new_t_end)
                
                row[col_start] = format_dt(final_start)
                row[col_end] = format_dt(new_t_end)

        final_finish = max_finish
            
        wait_delta = final_start - ready
        form_delta = timedelta(minutes=form_mins)
        tot_dur = (final_finish - arrival) + form_delta
        
        demurrage = max(timedelta(0), tot_dur - timedelta(hours=free_time_hours))
        
        row['Tippler Start Time'] = format_dt(final_start)
        row['Finish Unload'] = format_dt(final_finish)
        row['Wait (Tippler)'] = format_duration_hhmm(wait_delta)
        row['Total Duration'] = format_duration_hhmm(tot_dur)
        row['Demurrage'] = format_duration_hhmm(demurrage)
        
        date_str = arrival.strftime('%Y-%m-%d')
        daily_demurrage_tracker[date_str] = daily_demurrage_tracker.get(date_str, 0) + demurrage.total_seconds()
        
        recalc_rows.append(row)
        
    return pd.DataFrame(recalc_rows), daily_demurrage_tracker

# ==========================================
# 5. SIDEBAR PARAMS
# ==========================================
st.sidebar.header("⚙️ Settings")

# GOOGLE SHEET INPUT - HARDCODED DEFAULT
st.sidebar.subheader("🌍 Data Source")
gs_url = st.sidebar.text_input(
    "Google Sheet CSV Link", 
    value="https://docs.google.com/spreadsheets/d/e/2PACX-1vTlqPtwJyVkJYLs3V2t1kMw0It1zURfH3fU7vtLKX0BaQ_p71b2xvkH4NRazgD9Bg/pub?output=csv"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Infrastructure")
sim_params = {}
sim_params['rt1'] = st.sidebar.number_input("Tippler 1 Rate", value=6.0, step=0.5)
sim_params['rt2'] = st.sidebar.number_input("Tippler 2 Rate", value=6.0, step=0.5)
sim_params['rt3'] = st.sidebar.number_input("Tippler 3 Rate", value=9.0, step=0.5)
sim_params['rt4'] = st.sidebar.number_input("Tippler 4 Rate", value=9.0, step=0.5)
st.sidebar.markdown("---")
sim_params['sa'] = st.sidebar.number_input("Pair A Shunt (Mins)", value=25.0, step=5.0)
sim_params['sb'] = st.sidebar.number_input("Pair B Shunt (Mins)", value=50.0, step=5.0)
sim_params['extra_shunt'] = st.sidebar.number_input("Cross-Pair Penalty (Mins)", value=45.0, step=5.0)
st.sidebar.markdown("---")
sim_params['fa'] = st.sidebar.number_input("Pair A Formation (Mins)", value=20.0, step=5.0)
sim_params['fb'] = st.sidebar.number_input("Pair B Formation (Mins)", value=50.0, step=5.0)
sim_params['ft'] = st.sidebar.number_input("Free Time (Hours)", value=9.0, step=0.5)
sim_params['wb'] = st.sidebar.number_input("Wagons 1st Batch", value=30, step=1)
sim_params['wd'] = st.sidebar.number_input("Delay 2nd Tippler (Mins)", value=0.0, step=5.0)

# DOWNTIME MANAGER
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Tippler Downtime")
if 'downtimes' not in st.session_state: st.session_state.downtimes = []
with st.sidebar.form("downtime_form"):
    dt_tippler = st.selectbox("Select Tippler", ["T1", "T2", "T3", "T4"])
    now_ist = datetime.now(IST)
    dt_start_date = st.date_input("Start Date", value=now_ist.date())
    dt_start_time = st.time_input("Start Time", value=now_ist.time())
    dt_duration = st.number_input("Duration (Minutes)", min_value=15, step=15, value=60)
    add_dt = st.form_submit_button("Add Downtime")
    if add_dt:
        start_naive = datetime.combine(dt_start_date, dt_start_time)
        start_dt = IST.localize(start_naive)
        end_dt = start_dt + timedelta(minutes=dt_duration)
        st.session_state.downtimes.append({"Tippler": dt_tippler, "Start": start_dt, "End": end_dt})
        st.rerun()

if st.session_state.downtimes:
    dt_df = pd.DataFrame(st.session_state.downtimes)
    display_df = dt_df.copy()
    display_df['Start'] = display_df['Start'].dt.strftime('%d-%H:%M')
    display_df['End'] = display_df['End'].dt.strftime('%d-%H:%M')
    st.sidebar.dataframe(display_df[['Tippler', 'Start', 'End']], use_container_width=True)
    if st.sidebar.button("Clear Downtimes"):
        st.session_state.downtimes = []
        st.rerun()

sim_params['downtimes'] = st.session_state.downtimes

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

uploaded_file = st.file_uploader("Upload FOIS CSV File (Plan)", type=["csv"])

# Trigger logic if either Source changes
input_changed = False
if uploaded_file is not None:
    if 'last_file_id' not in st.session_state or st.session_state.last_file_id != uploaded_file.file_id:
        input_changed = True
        st.session_state.last_file_id = uploaded_file.file_id

if gs_url and ('last_gs_url' not in st.session_state or st.session_state.last_gs_url != gs_url):
    input_changed = True
    st.session_state.last_gs_url = gs_url

if input_changed or 'raw_data_cached' not in st.session_state:
    # 1. Load Google Sheet Actuals
    actuals_df = pd.DataFrame()
    actuals_state = {}
    if gs_url:
        with st.spinner("Fetching Google Sheet Actuals..."):
            actuals_df, actuals_state = fetch_google_sheet_actuals(gs_url)
            if not actuals_df.empty:
                st.success(f"✅ Loaded {len(actuals_df)} actual rakes from Google Sheet (Today).")
    
    st.session_state.actuals_df = actuals_df
    st.session_state.actuals_state = actuals_state

    # 2. Load CSV Plan
    if uploaded_file:
        try:
            uploaded_file.seek(0) 
            df_raw = pd.read_csv(uploaded_file)
            df_raw.columns = df_raw.columns.str.strip().str.upper()
            if 'PLCT RESN' in df_raw.columns:
                df_raw = df_raw[~df_raw['PLCT RESN'].astype(str).str.upper().str.contains('LDNG', na=False)]
            st.session_state.raw_data_cached = df_raw
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.session_state.raw_data_cached = pd.DataFrame()

# Run Simulation if we have data
if 'raw_data_cached' in st.session_state or 'actuals_df' in st.session_state:
    df_raw = st.session_state.get('raw_data_cached', pd.DataFrame())
    df_act = st.session_state.get('actuals_df', pd.DataFrame())
    st_act = st.session_state.get('actuals_state', {})
    
    sim_result, sim_start_dt = run_full_simulation_initial(df_raw, sim_params, df_act, st_act)
    
    st.session_state.sim_result = sim_result
    st.session_state.sim_start_dt = sim_start_dt

    st.markdown("### 📝 Master Schedule (Actuals + Forecast)")
    
    if st.session_state.sim_result.empty:
        st.warning("No data found (Upload CSV or check Google Sheet link).")
    else:
        # Define Columns
        column_config = {
            "_Arrival_DT": None, "_Shunt_Ready_DT": None, "_Form_Mins": None,
            "Rake": st.column_config.TextColumn("Rake Name", disabled=True),
            "Status": st.column_config.TextColumn("Status", help="ACTUAL = From G-Sheet"),
            "Tippler Start Time": st.column_config.TextColumn("Start (dd-HH:MM)", help="Edit to delay"),
            "Finish Unload": st.column_config.TextColumn("Finish (dd-HH:MM)", help="Edit to extend"),
            "Extra Shunt (Mins)": st.column_config.NumberColumn("Ext. Shunt", step=5),
            "Optimization Type": st.column_config.TextColumn("Strategy", width="small"),
            "T1 Start": st.column_config.TextColumn("T1 Start"), "T1 End": st.column_config.TextColumn("T1 End"), "T1 Idle": st.column_config.TextColumn("T1 Idle"),
            "T2 Start": st.column_config.TextColumn("T2 Start"), "T2 End": st.column_config.TextColumn("T2 End"), "T2 Idle": st.column_config.TextColumn("T2 Idle"),
            "T3 Start": st.column_config.TextColumn("T3 Start"), "T3 End": st.column_config.TextColumn("T3 End"), "T3 Idle": st.column_config.TextColumn("T3 Idle"),
            "T4 Start": st.column_config.TextColumn("T4 Start"), "T4 End": st.column_config.TextColumn("T4 End"), "T4 Idle": st.column_config.TextColumn("T4 Idle"),
        }
        
        # Color highlighting for Actuals vs Simulated? 
        # Streamlit data_editor doesn't support row styling easily, but 'Status' column helps.
        
        edited_df = st.data_editor(
            st.session_state.sim_result,
            use_container_width=True,
            num_rows="fixed",
            column_config=column_config,
            disabled=["Rake", "Load Type", "Wagons", "Line Allotted", "Wait (Tippler)", "Total Duration", "Demurrage", "Tipplers Used", "Status"],
            key="data_editor"
        )

        # Reactive Calculation
        final_result, daily_stats = recalculate_cascade_reactive(edited_df, sim_params['ft'], st.session_state.sim_start_dt)
        
        # --- DEMURRAGE SUMMARY ---
        st.markdown("### 📅 Demurrage Forecast")
        col1, col2 = st.columns([1, 2])
        
        with col1:
             st.dataframe(pd.DataFrame(list(daily_stats.items()), columns=['Date', 'Seconds']).assign(
                 Demurrage=lambda x: x['Seconds'].apply(lambda s: format_duration_hhmm(timedelta(seconds=s)))
             )[['Date', 'Demurrage']], hide_index=True)

        # Download
        csv = final_result.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins"]).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Final Report", csv, "optimized_schedule.csv", "text/csv")
