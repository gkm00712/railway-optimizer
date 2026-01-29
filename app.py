import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import pytz
import math
import numpy as np
import re

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer (IST)", layout="wide")
st.title("🚂 BOXN Rake Demurrage Optimization Dashboard (IST)")

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
        val_str = str(val).strip()
        if '+' in val_str:
            parts = val_str.split('+')
            return sum(int(p) for p in parts if p.strip().isdigit())
        return int(float(val_str))
    except: return 0

def format_dt(dt):
    if pd.isnull(dt): return ""
    if dt.tzinfo is None: dt = IST.localize(dt)
    return dt.strftime('%d-%H:%M')

def restore_dt(dt_str, ref_dt):
    if not isinstance(dt_str, str) or dt_str.strip() == "": return pd.NaT
    try:
        parts = dt_str.split('-') 
        day = int(parts[0])
        time_parts = parts[1].split(':')
        hour, minute = int(time_parts[0]), int(time_parts[1])
        
        if ref_dt.tzinfo is None: ref_dt = IST.localize(ref_dt)
        new_dt = ref_dt.replace(day=day, hour=hour, minute=minute, second=0)
        
        if day < ref_dt.day - 15: new_dt = new_dt + pd.DateOffset(months=1)
        elif day > ref_dt.day + 15: new_dt = new_dt - pd.DateOffset(months=1)
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

def calculate_rounded_demurrage(duration_delta, free_time_hours):
    """
    1. Subtract Free Time from Total Duration.
    2. If result <= 0, Demurrage = 0.
    3. If result > 0, Round UP to nearest whole hour (Ceiling).
    """
    if pd.isnull(duration_delta): return "00:00", 0
    
    # Total minutes taken
    total_minutes = duration_delta.total_seconds() / 60
    free_minutes = free_time_hours * 60
    
    demurrage_minutes = total_minutes - free_minutes
    
    if demurrage_minutes <= 0:
        return "00:00", 0
    
    # Ceiling logic: 1 min -> 1 hr, 61 mins -> 2 hrs
    rounded_hours = math.ceil(demurrage_minutes / 60)
    
    return f"{int(rounded_hours):02d}:00", rounded_hours

def check_downtime_impact(tippler_id, proposed_start, downtime_list):
    if proposed_start.tzinfo is None: proposed_start = IST.localize(proposed_start)
    relevant_dts = [d for d in downtime_list if d['Tippler'] == tippler_id]
    relevant_dts.sort(key=lambda x: x['Start'])
    
    current_start = proposed_start
    changed = True
    while changed:
        changed = False
        for dt in relevant_dts:
            if dt['Start'] <= current_start < dt['End']:
                current_start = dt['End']
                changed = True
    return current_start

def find_column(df, candidates):
    """Smartly looks for a column matching one of the candidates."""
    cols_upper = [str(c).upper().strip() for c in df.columns]
    for cand in candidates:
        cand_upper = cand.upper().strip()
        if cand_upper in cols_upper:
            return df.columns[cols_upper.index(cand_upper)]
        for c in cols_upper:
            if cand_upper == c: 
                 return df.columns[cols_upper.index(c)]
    return None

def is_valid_demurrage_string(val):
    """
    Checks if the value is a valid number or time format.
    Returns True if valid (e.g. "2", "2.5", "02:00"), False if text/garbage.
    """
    s = str(val).strip()
    if not s or s.lower() == 'nan': return False
    
    # Check for HH:MM format
    if re.match(r'^\d{1,2}:\d{2}$', s): return True
    
    # Check for pure number (int or float)
    try:
        float(s)
        return True
    except:
        return False

# ==========================================
# 3. GOOGLE SHEET PARSER (Cached & Strict)
# ==========================================

def safe_parse_date(val):
    if pd.isnull(val) or str(val).strip() == "" or str(val).strip().upper() == "U/P": return pd.NaT
    try:
        dt = pd.to_datetime(val, dayfirst=True, errors='coerce') 
        if pd.isnull(dt): return pd.NaT 
        return to_ist(dt)
    except:
        return pd.NaT

@st.cache_data(ttl=60)
def fetch_google_sheet_actuals(url, free_time_hours):
    try:
        # Read twice: once to get headers, once to get data by index
        df_headers = pd.read_csv(url, nrows=0) 
        df_gs = pd.read_csv(url, header=None, skiprows=1) # Data only
        
        # Identify Demurrage Column by Name if it exists
        dem_col_idx = -1
        cols_upper = [str(c).upper().strip() for c in df_headers.columns]
        for i, c in enumerate(cols_upper):
            if "DEMURRAGE" in c or "WHARFAGE" in c:
                dem_col_idx = i
                break

        if len(df_gs.columns) < 18:
            return pd.DataFrame(), {}

        # STRICT COLUMN MAPPING (User Rules):
        # Col B (Idx 1) = Rake Name
        # Col C (Idx 2) = Source/Mine (Strictly requested)
        # Col E (Idx 4) = Receipt
        # Col F (Idx 5) = Placement
        # Col G (Idx 6) = Unload End
        # Col H (Idx 7) = Release
        # Col I (Idx 8) = Total Duration
        # Col O-R (Idx 14-17) = Tipplers
        
        actuals = []
        today_date = datetime.now(IST).date()

        for _, row in df_gs.iterrows():
            arrival_dt = safe_parse_date(row.iloc[4]) # Col E
            if pd.isnull(arrival_dt): continue

            # Filter: ONLY TODAY'S DATA
            if arrival_dt.date() != today_date: continue

            rake_name = str(row.iloc[1]) # Col B
            source_val = str(row.iloc[2]) # Col C (Fixed as per request)
            if source_val.lower() == 'nan': source_val = ""

            start_dt = safe_parse_date(row.iloc[5])   # Col F
            end_dt = safe_parse_date(row.iloc[6])     # Col G
            
            # Duration - Try Column I (Index 8)
            raw_dur = row.iloc[8]
            total_dur = timedelta(0)
            
            if pd.notnull(raw_dur):
                try:
                    if ":" in str(raw_dur):
                        parts = str(raw_dur).split(":")
                        total_dur = timedelta(hours=int(parts[0]), minutes=int(parts[1]))
                    else:
                        days = float(raw_dur)
                        total_dur = timedelta(days=days)
                except:
                    # Fallback to Calc
                    release_dt = safe_parse_date(row.iloc[7])
                    if pd.notnull(release_dt):
                        total_dur = release_dt - arrival_dt

            if pd.isnull(start_dt): start_dt = arrival_dt 

            # Demurrage Logic:
            # 1. Check if column exists AND value is valid number/time
            use_sheet_demurrage = False
            dem_str = "00:00"
            
            if dem_col_idx != -1:
                raw_dem_val = row.iloc[dem_col_idx]
                if is_valid_demurrage_string(raw_dem_val):
                    use_sheet_demurrage = True
                    dem_str = str(raw_dem_val).strip()
                    # Standardize "2" -> "02:00" if possible
                    try:
                        if ":" not in dem_str:
                            dem_str = f"{int(float(dem_str)):02d}:00"
                    except: pass
            
            # 2. If invalid or missing, Calculate it
            if not use_sheet_demurrage:
                dem_str, _ = calculate_rounded_demurrage(total_dur, free_time_hours)

            # Tipplers
            used = []
            if pd.notnull(row.iloc[14]) and str(row.iloc[14]).strip() not in ["", "nan"]: used.append("T1")
            if pd.notnull(row.iloc[15]) and str(row.iloc[15]).strip() not in ["", "nan"]: used.append("T2")
            if pd.notnull(row.iloc[16]) and str(row.iloc[16]).strip() not in ["", "nan"]: used.append("T3")
            if pd.notnull(row.iloc[17]) and str(row.iloc[17]).strip() not in ["", "nan"]: used.append("T4")
            
            entry = {
                'Rake': rake_name,
                'Coal Source': source_val,
                'Load Type': 'BOXN (Actual)',
                'Wagons': 58,
                'Status': 'ACTUAL',
                '_Arrival_DT': arrival_dt,
                '_Shunt_Ready_DT': start_dt,
                '_Form_Mins': 0,
                'Optimization Type': 'Actual (G-Sheet)',
                'Extra Shunt (Mins)': 0,
                'Line Allotted': 'N/A',
                'Line Entry Time': format_dt(arrival_dt),
                'Shunting Complete': format_dt(start_dt),
                'Tippler Start Time': format_dt(start_dt),
                'Finish Unload': format_dt(end_dt),
                'Tipplers Used': ", ".join(used),
                'Wait (Tippler)': format_duration_hhmm(start_dt - arrival_dt) if pd.notnull(start_dt) else "",
                'Total Duration': format_duration_hhmm(total_dur),
                'Demurrage': dem_str,
                '_raw_end_dt': end_dt,
                '_raw_tipplers': used
            }
            
            for t in ['T1', 'T2', 'T3', 'T4']:
                entry[f"{t} Start"] = format_dt(start_dt) if t in used else ""
                entry[f"{t} End"] = format_dt(end_dt) if t in used else ""
                entry[f"{t} Idle"] = ""
            
            actuals.append(entry)

        df_actuals = pd.DataFrame(actuals)
        
        tippler_init_state = {}
        for t in ['T1', 'T2', 'T3', 'T4']:
            t_max = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
            if not df_actuals.empty:
                for _, row in df_actuals.iterrows():
                    if t in row['_raw_tipplers'] and pd.notnull(row['_raw_end_dt']):
                        if row['_raw_end_dt'] > t_max: t_max = row['_raw_end_dt']
            tippler_init_state[t] = t_max

        return df_actuals, tippler_init_state

    except Exception as e:
        return pd.DataFrame(), {}

# ==========================================
# 4. CORE SIMULATION LOGIC
# ==========================================

def calculate_generic_finish(wagons, target_tipplers, ready_time, tippler_state, downtime_list, 
                           rates, wagons_first_batch, inter_tippler_delay):
    if wagons == 0: return ready_time, "", ready_time, {}, timedelta(0)
    
    resources = {t: rates[t] for t in target_tipplers}
    sorted_tipplers = sorted(resources.keys(), key=lambda x: tippler_state[x])
    
    t_primary = sorted_tipplers[0]
    t_secondary = sorted_tipplers[1] if len(sorted_tipplers) > 1 else None
    
    detailed_timings = {}
    used_tipplers_list = []
    
    w_first = min(wagons, wagons_first_batch)
    w_second = wagons - w_first
    
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

    return overall_finish, ", ".join(sorted(used_tipplers_list)), actual_start_prim, detailed_timings, total_idle_cost

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
    rates = {'T1': params['rt1'], 'T2': params['rt2'], 'T3': params['rt3'], 'T4': params['rt4']}
    s_a, s_b, extra_shunt_cross = params['sa'], params['sb'], params['extra_shunt']
    f_a, f_b, ft_hours = params['fa'], params['fb'], params['ft']
    w_batch, w_delay, downtimes = params['wb'], params['wd'], params['downtimes']

    df = df.copy()
    
    # 1. Load Type
    load_col = find_column(df, ['LOAD TYPE', 'CMDT', 'COMMODITY'])
    if load_col:
        df = df[df[load_col].astype(str).str.strip().str.upper().str.startswith('BOXN')]
    
    # 2. Wagon Count
    wagon_col = find_column(df, ['TOTL UNTS', 'WAGONS', 'UNITS', 'TOTAL UNITS', 'NO OF WAGONS'])
    if wagon_col:
        df['wagon_count'] = df[wagon_col].apply(parse_wagons)
    else:
        df['wagon_count'] = 58 

    # 3. Rake Name
    rake_col = find_column(df, ['RAKE NAME', 'RAKE', 'TRAIN NAME'])
    if not rake_col: return pd.DataFrame(), None 
    
    # 4. Source Column (CSV) - Prioritize STTS FROM
    src_col = find_column(df, ['STTS FROM', 'STTN FROM', 'FROM_STN', 'SRC', 'SOURCE', 'FROM'])
        
    # 5. Arrival Time
    arvl_col = find_column(df, ['EXPD ARVLTIME', 'ARRIVAL TIME', 'EXPECTED ARRIVAL'])
    stts_time_col = find_column(df, ['STTS TIME'])
    stts_code_col = find_column(df, ['STTS CODE', 'STATUS'])

    df['exp_arrival_dt'] = pd.to_datetime(df[arvl_col], errors='coerce').apply(to_ist) if arvl_col else pd.NaT
    
    if stts_time_col and stts_code_col:
        df['stts_time_dt'] = pd.to_datetime(df[stts_time_col], errors='coerce').apply(to_ist)
        df['arrival_dt'] = df.apply(lambda r: r['stts_time_dt'] if str(r.get(stts_code_col)).strip()=='PL' and pd.notnull(r['stts_time_dt']) else r['exp_arrival_dt'], axis=1)
    else:
        df['arrival_dt'] = df['exp_arrival_dt']

    df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt').reset_index(drop=True)

    # DEDUPLICATE against Actuals
    if not actuals_df.empty:
        actual_rakes = actuals_df['Rake'].astype(str).tolist()
        df = df[~df[rake_col].astype(str).isin(actual_rakes)]

    if df.empty and actuals_df.empty: return pd.DataFrame(), None

    # INITIALIZE STATE
    first_arrival = df['arrival_dt'].min() if not df.empty else datetime.now(IST)
    sim_start_time = first_arrival.replace(hour=0, minute=0, second=0, microsecond=0)
    if sim_start_time.tzinfo is None: sim_start_time = IST.localize(sim_start_time)

    tippler_state = actuals_state if actuals_state else {k: sim_start_time for k in ['T1', 'T2', 'T3', 'T4']}
    spec_line_status = {8: sim_start_time, 9: sim_start_time, 10: sim_start_time, 11: sim_start_time}
    line_groups = {
        'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}, 
        'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}
    }
    rr_tracker_A = -1 
    assignments = []
    
    for _, rake in df.iterrows():
        options = []
        # Create 4 options logic
        entry_A = get_line_entry_time('Group_Lines_8_10', rake['arrival_dt'], line_groups)
        ready_A = entry_A + timedelta(minutes=s_a)
        fin_A, used_A, start_A, tim_A, idle_A = calculate_generic_finish(
            rake['wagon_count'], ['T1', 'T2'], ready_A, tippler_state, downtimes, rates, w_batch, w_delay)
        
        dur_A = (fin_A - rake['arrival_dt']) + timedelta(minutes=f_a)
        dem_A_raw = max(0, (dur_A - timedelta(hours=ft_hours)).total_seconds())

        options.append({'id': 'Nat_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_A, 'start': start_A, 'fin': fin_A, 'used': used_A, 'timings': tim_A, 'idle': idle_A, 'dem_raw': dem_A_raw, 'extra_shunt': 0.0, 'form_mins': f_a, 'type': 'Standard'})

        entry_B = get_line_entry_time('Group_Line_11', rake['arrival_dt'], line_groups)
        ready_B = entry_B + timedelta(minutes=s_b)
        fin_B, used_B, start_B, tim_B, idle_B = calculate_generic_finish(
            rake['wagon_count'], ['T3', 'T4'], ready_B, tippler_state, downtimes, rates, w_batch, w_delay)
        dur_B = (fin_B - rake['arrival_dt']) + timedelta(minutes=f_b)
        dem_B_raw = max(0, (dur_B - timedelta(hours=ft_hours)).total_seconds())
        
        options.append({'id': 'Nat_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_B, 'start': start_B, 'fin': fin_B, 'used': used_B, 'timings': tim_B, 'idle': idle_B, 'dem_raw': dem_B_raw, 'extra_shunt': 0.0, 'form_mins': f_b, 'type': 'Standard'})

        ready_A_Cross = ready_A + timedelta(minutes=extra_shunt_cross)
        fin_AX, used_AX, start_AX, tim_AX, idle_AX = calculate_generic_finish(
            rake['wagon_count'], ['T3', 'T4'], ready_A_Cross, tippler_state, downtimes, rates, w_batch, w_delay)
        dur_AX = (fin_AX - rake['arrival_dt']) + timedelta(minutes=f_a)
        dem_AX_raw = max(0, (dur_AX - timedelta(hours=ft_hours)).total_seconds())
        
        options.append({'id': 'Cross_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_A_Cross, 'start': start_AX, 'fin': fin_AX, 'used': used_AX, 'timings': tim_AX, 'idle': idle_AX, 'dem_raw': dem_AX_raw, 'extra_shunt': float(extra_shunt_cross), 'form_mins': f_a, 'type': 'Cross-Transfer'})

        ready_B_Cross = ready_B + timedelta(minutes=extra_shunt_cross)
        fin_BX, used_BX, start_BX, tim_BX, idle_BX = calculate_generic_finish(
            rake['wagon_count'], ['T1', 'T2'], ready_B_Cross, tippler_state, downtimes, rates, w_batch, w_delay)
        dur_BX = (fin_BX - rake['arrival_dt']) + timedelta(minutes=f_b)
        dem_BX_raw = max(0, (dur_BX - timedelta(hours=ft_hours)).total_seconds())
        
        options.append({'id': 'Cross_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_B_Cross, 'start': start_BX, 'fin': fin_BX, 'used': used_BX, 'timings': tim_BX, 'idle': idle_BX, 'dem_raw': dem_BX_raw, 'extra_shunt': float(extra_shunt_cross), 'form_mins': f_b, 'type': 'Cross-Transfer'})

        # Sort by Demurrage Cost -> Finish Time -> Idle
        best_opt = sorted(options, key=lambda x: (x['dem_raw'], x['fin'], x['idle']))[0]

        for k, v in best_opt['timings'].items():
            if 'End' in k: tippler_state[k.split('_')[0]] = v
        
        if best_opt['grp'] == 'Group_Lines_8_10':
            sel_line = find_specific_line(best_opt['grp'], best_opt['entry'], spec_line_status, rr_tracker_A)
            if sel_line in [8,9,10]: rr_tracker_A = {8:0,9:1,10:2}[sel_line]
        else: sel_line = 11
        
        line_groups[best_opt['grp']]['line_free_times'].append(best_opt['entry'] + timedelta(minutes=line_groups[best_opt['grp']]['clearance_mins']))
        spec_line_status[sel_line] = line_groups[best_opt['grp']]['line_free_times'][-1]

        # Final Rounded Calculation for Display
        tot_dur_final = (best_opt['fin'] - rake['arrival_dt']) + timedelta(minutes=best_opt['form_mins'])
        dem_str, _ = calculate_rounded_demurrage(tot_dur_final, ft_hours)

        row_data = {
            'Rake': rake[rake_col],
            'Coal Source': rake.get(src_col, '') if src_col else '',
            'Load Type': rake.get(load_col, 'BOXN'),
            'Wagons': rake['wagon_count'],
            'Status': rake.get(stts_code_col, 'N/A') if stts_code_col else 'N/A',
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
            'Wait (Tippler)': format_duration_hhmm(best_opt['start'] - best_opt['ready']),
            'Total Duration': format_duration_hhmm(tot_dur_final),
            'Demurrage': dem_str
        }
        for t in ['T1', 'T2', 'T3', 'T4']:
             row_data[f"{t} Start"] = format_dt(best_opt['timings'].get(f"{t}_Start", pd.NaT))
             row_data[f"{t} End"] = format_dt(best_opt['timings'].get(f"{t}_End", pd.NaT))
             row_data[f"{t} Idle"] = format_duration_hhmm(best_opt['timings'].get(f"{t}_Idle", pd.NaT))
        assignments.append(row_data)

    df_sim = pd.DataFrame(assignments)
    
    if not actuals_df.empty:
        actuals_clean = actuals_df.drop(columns=['_raw_end_dt', '_raw_tipplers'], errors='ignore')
        final_df = pd.concat([actuals_clean, df_sim], ignore_index=True) if not df_sim.empty else actuals_clean
    else:
        final_df = df_sim

    return final_df, sim_start_time

def recalculate_cascade_reactive(edited_df, free_time_hours, sim_start_dt):
    recalc_rows = []
    if sim_start_dt.tzinfo is None: sim_start_dt = IST.localize(sim_start_dt)
    tippler_state = {k: sim_start_dt for k in ['T1', 'T2', 'T3', 'T4']}
    daily_demurrage_hours = {}
    
    for _, row in edited_df.iterrows():
        # Handle Actuals
        if row.get('Status') == 'ACTUAL':
            for t in ['T1', 'T2', 'T3', 'T4']:
                end_str = row.get(f"{t} End")
                if pd.notnull(end_str) and end_str != "":
                    end_dt = restore_dt(end_str, sim_start_dt)
                    if pd.notnull(end_dt) and end_dt > tippler_state[t]: tippler_state[t] = end_dt
            
            # Use existing Demurrage column for aggregation if valid
            try:
                dem_val = row['Demurrage']
                dem_hrs = 0
                if ":" in str(dem_val):
                    dem_hrs = int(dem_val.split(":")[0])
                elif pd.notnull(dem_val):
                    # Try converting if it's just a number
                    try: dem_hrs = int(float(dem_val))
                    except: 
                        # Fallback calc if not a direct number/time
                        dur_str = row['Total Duration']
                        sign = -1 if dur_str.startswith('-') else 1
                        parts = dur_str.replace('-','').split(':')
                        dur_td = timedelta(hours=int(parts[0]), minutes=int(parts[1])) * sign
                        _, dem_hrs = calculate_rounded_demurrage(dur_td, free_time_hours)
            except: dem_hrs = 0
            
            try:
                arr_dt = pd.to_datetime(row['_Arrival_DT'])
                if arr_dt.tzinfo is None: arr_dt = IST.localize(arr_dt)
                d_str = arr_dt.strftime('%Y-%m-%d')
                daily_demurrage_hours[d_str] = daily_demurrage_hours.get(d_str, 0) + dem_hrs
            except: pass
            
            recalc_rows.append(row)
            continue
            
        # Handle Simulated
        arrival = pd.to_datetime(row['_Arrival_DT'])
        if arrival.tzinfo is None: arrival = IST.localize(arrival)
        ready = pd.to_datetime(row['_Shunt_Ready_DT'])
        if ready.tzinfo is None: ready = IST.localize(ready)
        
        used_str = str(row['Tipplers Used'])
        current_tipplers = [t for t in ['T1', 'T2', 'T3', 'T4'] if t in used_str]
        
        resource_free_at = sim_start_dt
        for t in current_tipplers: resource_free_at = max(resource_free_at, tippler_state[t])
        
        valid_start = max(ready, resource_free_at)
        user_start = restore_dt(row['Tippler Start Time'], ready)
        final_start = max(valid_start, user_start) if pd.notnull(user_start) else valid_start
        max_finish = final_start 
        
        for t_id in ['T1', 'T2', 'T3', 'T4']:
            if t_id in current_tipplers:
                row[f"{t_id} Idle"] = format_duration_hhmm(max(timedelta(0), final_start - tippler_state[t_id]))
                
                t_end_val = restore_dt(row[f"{t_id} End"], final_start)
                t_start_val = restore_dt(row[f"{t_id} Start"], final_start)
                duration = t_end_val - t_start_val if (pd.notnull(t_end_val) and pd.notnull(t_start_val)) else timedelta(hours=2)
                if duration < timedelta(0): duration = timedelta(hours=2)

                new_t_end = final_start + duration
                if pd.notnull(t_end_val) and t_end_val > new_t_end: new_t_end = t_end_val

                tippler_state[t_id] = new_t_end
                max_finish = max(max_finish, new_t_end)
                row[f"{t_id} Start"] = format_dt(final_start)
                row[f"{t_id} End"] = format_dt(new_t_end)

        tot_dur = (max_finish - arrival) + timedelta(minutes=float(row['_Form_Mins']))
        dem_str, dem_hrs = calculate_rounded_demurrage(tot_dur, free_time_hours)
        
        row['Tippler Start Time'] = format_dt(final_start)
        row['Finish Unload'] = format_dt(max_finish)
        row['Wait (Tippler)'] = format_duration_hhmm(final_start - ready)
        row['Total Duration'] = format_duration_hhmm(tot_dur)
        row['Demurrage'] = dem_str
        
        d_str = arrival.strftime('%Y-%m-%d')
        daily_demurrage_hours[d_str] = daily_demurrage_hours.get(d_str, 0) + dem_hrs
        recalc_rows.append(row)
        
    return pd.DataFrame(recalc_rows), daily_demurrage_hours

# ==========================================
# 5. SIDEBAR PARAMS
# ==========================================
st.sidebar.header("⚙️ Settings")
gs_url = st.sidebar.text_input("Google Sheet CSV Link", value="https://docs.google.com/spreadsheets/d/e/2PACX-1vTlqPtwJyVkJYLs3V2t1kMw0It1zURfH3fU7vtLKX0BaQ_p71b2xvkH4NRazgD9Bg/pub?output=csv")

st.sidebar.markdown("---")
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

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Tippler Downtime")
if 'downtimes' not in st.session_state: st.session_state.downtimes = []
with st.sidebar.form("downtime_form"):
    dt_tippler = st.selectbox("Select Tippler", ["T1", "T2", "T3", "T4"])
    now_ist = datetime.now(IST)
    dt_start_date = st.date_input("Start Date", value=now_ist.date())
    dt_start_time = st.time_input("Start Time", value=now_ist.time())
    dt_duration = st.number_input("Duration (Minutes)", min_value=15, step=15, value=60)
    if st.form_submit_button("Add Downtime"):
        start_dt = IST.localize(datetime.combine(dt_start_date, dt_start_time))
        st.session_state.downtimes.append({"Tippler": dt_tippler, "Start": start_dt, "End": start_dt + timedelta(minutes=dt_duration)})
        st.rerun()

if st.session_state.downtimes:
    dt_df = pd.DataFrame(st.session_state.downtimes)
    st.sidebar.dataframe(dt_df.assign(Start=lambda x: x['Start'].dt.strftime('%d-%H:%M'), End=lambda x: x['End'].dt.strftime('%d-%H:%M'))[['Tippler', 'Start', 'End']], use_container_width=True)
    if st.sidebar.button("Clear Downtimes"):
        st.session_state.downtimes = []
        st.rerun()
sim_params['downtimes'] = st.session_state.downtimes

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

# 1. LIVE GOOGLE SHEET DISPLAY (BEFORE UPLOAD)
if gs_url and 'actuals_df_disp' not in st.session_state:
    st.session_state.actuals_df_disp = pd.DataFrame()
    
if gs_url:
    try:
        if 'actuals_df' not in st.session_state or st.session_state.actuals_df.empty:
             df_disp, _ = fetch_google_sheet_actuals(gs_url, sim_params['ft'])
             st.session_state.actuals_df_disp = df_disp
        else:
             st.session_state.actuals_df_disp = st.session_state.actuals_df
    except: pass

st.subheader("📊 Live Unloading Status (Today)")
if 'actuals_df_disp' in st.session_state and not st.session_state.actuals_df_disp.empty:
    disp_cols = ['Rake', 'Coal Source', 'Status', 'Tippler Start Time', 'Finish Unload', 'Total Duration', 'Demurrage']
    st.dataframe(st.session_state.actuals_df_disp[disp_cols], use_container_width=True)
else:
    st.info("Loading Google Sheet data or no data found for today...")

# 2. FILE UPLOAD & PROCESSING
uploaded_file = st.file_uploader("Upload FOIS CSV File (Plan)", type=["csv"])

input_changed = False
if uploaded_file and ('last_file_id' not in st.session_state or st.session_state.last_file_id != uploaded_file.file_id):
    input_changed = True
    st.session_state.last_file_id = uploaded_file.file_id
if gs_url and ('last_gs_url' not in st.session_state or st.session_state.last_gs_url != gs_url):
    input_changed = True
    st.session_state.last_gs_url = gs_url

if input_changed or 'raw_data_cached' not in st.session_state:
    actuals_df, actuals_state = pd.DataFrame(), {}
    if gs_url:
        actuals_df, actuals_state = fetch_google_sheet_actuals(gs_url, sim_params['ft'])
        st.session_state.actuals_df_disp = actuals_df
    
    st.session_state.actuals_df = actuals_df
    st.session_state.actuals_state = actuals_state

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
    else: st.session_state.raw_data_cached = pd.DataFrame()

if 'raw_data_cached' in st.session_state or 'actuals_df' in st.session_state:
    df_raw = st.session_state.get('raw_data_cached', pd.DataFrame())
    df_act = st.session_state.get('actuals_df', pd.DataFrame())
    st_act = st.session_state.get('actuals_state', {})
    
    sim_result, sim_start_dt = run_full_simulation_initial(df_raw, sim_params, df_act, st_act)
    st.session_state.sim_result = sim_result
    st.session_state.sim_start_dt = sim_start_dt

    if not st.session_state.sim_result.empty:
        col_cfg = {
            "Rake": st.column_config.TextColumn("Rake Name", disabled=True),
            "Coal Source": st.column_config.TextColumn("Source/Mine", disabled=True),
            "Status": st.column_config.TextColumn("Status", help="ACTUAL = From G-Sheet"),
            "Tippler Start Time": st.column_config.TextColumn("Start (dd-HH:MM)"),
            "Finish Unload": st.column_config.TextColumn("Finish (dd-HH:MM)"),
            "Extra Shunt (Mins)": st.column_config.NumberColumn("Ext. Shunt", step=5),
            "_Arrival_DT": None, "_Shunt_Ready_DT": None, "_Form_Mins": None
        }
        
        st.markdown("### 📝 Master Schedule (Actuals + Forecast)")
        edited_df = st.data_editor(
            st.session_state.sim_result, 
            use_container_width=True, 
            num_rows="fixed", 
            column_config=col_cfg, 
            disabled=["Rake", "Coal Source", "Load Type", "Wagons", "Line Allotted", "Wait (Tippler)", "Total Duration", "Demurrage", "Tipplers Used", "Status"],
            key="schedule_editor" 
        )
        
        final_result, daily_stats = recalculate_cascade_reactive(edited_df, sim_params['ft'], st.session_state.sim_start_dt)
        
        st.markdown("### 📅 Demurrage Forecast (Rounded to Next Hour)")
        st.dataframe(pd.DataFrame(list(daily_stats.items()), columns=['Date', 'Total Hours']).assign(Demurrage=lambda x: x['Total Hours'].apply(lambda h: f"{int(h)} Hours"))[['Date', 'Demurrage']], hide_index=True)
        
        st.download_button("📥 Download Final Report", final_result.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins"]).to_csv(index=False).encode('utf-8'), "optimized_schedule.csv", "text/csv")
