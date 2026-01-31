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
    
    total_minutes = duration_delta.total_seconds() / 60
    free_minutes = free_time_hours * 60
    demurrage_minutes = total_minutes - free_minutes
    
    if demurrage_minutes <= 0: return "00:00", 0
    
    # Ceiling logic
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
    s = str(val).strip()
    if not s or s.lower() == 'nan': return False
    if re.match(r'^\d{1,2}:\d{2}$', s): return True
    try:
        float(s)
        return True
    except:
        return False

def get_sequence_number(rake_name):
    """
    Extracts the leading number from ANY format.
    '148/BOXN' -> 148
    '148 BOXN' -> 148
    '148'      -> 148
    """
    try:
        # Match any digits at the start of the string
        match = re.search(r'^(\d+)', str(rake_name).strip())
        if match:
            return int(match.group(1))
    except: pass
    return 0

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
        df_headers = pd.read_csv(url, nrows=0) 
        df_gs = pd.read_csv(url, header=None, skiprows=1) # Data only
        
        # Identify Demurrage Column by Name if it exists
        dem_col_idx = -1
        cols_upper = [str(c).upper().strip() for c in df_headers.columns]
        for i, c in enumerate(cols_upper):
            if "DEMURRAGE" in c or "WHARFAGE" in c:
                dem_col_idx = i
                break

        if len(df_gs.columns) < 18: return pd.DataFrame(), pd.DataFrame(), 0

        locked_actuals = []
        unplanned_actuals = [] 
        max_seq_num = 0
        today_date = datetime.now(IST).date()

        for _, row in df_gs.iterrows():
            arrival_dt = safe_parse_date(row.iloc[4]) # Col E
            if pd.isnull(arrival_dt): continue
            if arrival_dt.date() != today_date: continue

            rake_name = str(row.iloc[1]) # Col B
            
            # --- Sequence Tracking Fix ---
            seq = get_sequence_number(rake_name)
            if seq > max_seq_num: max_seq_num = seq

            source_val = str(row.iloc[2]) # Col C (Strict)
            if source_val.lower() == 'nan': source_val = ""

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
                '_Form_Mins': 0,
                'Extra Shunt (Mins)': 0,
                'Line Allotted': 'N/A',
                'Line Entry Time': format_dt(arrival_dt)
            }

            if not used:
                # If no tipplers, send to optimizer
                entry['Status'] = 'Pending (G-Sheet)'
                entry['Optimization Type'] = 'Auto-Planned'
                unplanned_actuals.append(entry)
            else:
                start_dt = safe_parse_date(row.iloc[5])
                end_dt = safe_parse_date(row.iloc[6])
                if pd.isnull(start_dt): start_dt = arrival_dt
                
                # Duration & Demurrage
                raw_dur = row.iloc[8]
                total_dur = timedelta(0)
                if pd.notnull(raw_dur):
                    try:
                        if ":" in str(raw_dur):
                            parts = str(raw_dur).split(":")
                            total_dur = timedelta(hours=int(parts[0]), minutes=int(parts[1]))
                        else: total_dur = timedelta(days=float(raw_dur))
                    except:
                        if pd.notnull(safe_parse_date(row.iloc[7])): 
                            total_dur = safe_parse_date(row.iloc[7]) - arrival_dt

                # Demurrage Check
                use_sheet_dem = False
                dem_str = "00:00"
                if dem_col_idx != -1:
                    raw_dem = row.iloc[dem_col_idx]
                    if is_valid_demurrage_string(raw_dem):
                        use_sheet_dem = True
                        dem_str = str(raw_dem).strip()
                        if ":" not in dem_str: 
                            try: dem_str = f"{int(float(dem_str)):02d}:00"
                            except: pass
                
                if not use_sheet_dem: 
                    dem_str, _ = calculate_rounded_demurrage(total_dur, free_time_hours)

                entry.update({
                    '_Shunt_Ready_DT': start_dt,
                    'Optimization Type': 'Actual (G-Sheet)',
                    'Shunting Complete': format_dt(start_dt),
                    'Tippler Start Time': format_dt(start_dt),
                    'Finish Unload': format_dt(end_dt),
                    'Tipplers Used': ", ".join(used),
                    'Wait (Tippler)': format_duration_hhmm(start_dt - arrival_dt),
                    'Total Duration': format_duration_hhmm(total_dur),
                    'Demurrage': dem_str,
                    '_raw_end_dt': end_dt,
                    '_raw_tipplers': used
                })
                
                for t in ['T1', 'T2', 'T3', 'T4']:
                    entry[f"{t} Start"] = format_dt(start_dt) if t in used else ""
                    entry[f"{t} End"] = format_dt(end_dt) if t in used else ""
                    entry[f"{t} Idle"] = ""
                
                locked_actuals.append(entry)

        return pd.DataFrame(locked_actuals), pd.DataFrame(unplanned_actuals), max_seq_num

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), 0

# ==========================================
# 4. CORE SIMULATION LOGIC
# ==========================================

def calculate_generic_finish(wagons, target_tipplers, ready_time, tippler_state, downtime_list, 
                           rates, wagons_first_batch, inter_tippler_delay):
    if wagons == 0: return ready_time, "", ready_time, {}, timedelta(0)
    
    candidates = []
    for t in target_tipplers:
        free_at = tippler_state[t]
        if free_at.tzinfo is None: free_at = IST.localize(free_at)
        prop_start = max(ready_time, free_at)
        effective_start = check_downtime_impact(t, prop_start, downtime_list)
        candidates.append({'id': t, 'rate': rates[t], 'eff_start': effective_start, 'free_at': free_at})
    
    sorted_tipplers = sorted(candidates, key=lambda x: x['eff_start'])
    prim = sorted_tipplers[0]
    t_primary = prim['id']
    
    detailed_timings = {}
    used_tipplers_list = [t_primary]
    
    finish_A = prim['eff_start'] + timedelta(hours=wagons / prim['rate'])
    
    finish_B = pd.NaT
    sec = None
    if len(sorted_tipplers) > 1 and wagons > wagons_first_batch:
        sec = sorted_tipplers[1]
        w_first = wagons_first_batch
        w_second = wagons - w_first
        fin_prim_split = prim['eff_start'] + timedelta(hours=w_first / prim['rate'])
        start_sec_theory = ready_time + timedelta(minutes=inter_tippler_delay)
        prop_start_sec = max(start_sec_theory, sec['eff_start'])
        real_start_sec = check_downtime_impact(sec['id'], prop_start_sec, downtime_list)
        fin_sec_split = real_start_sec + timedelta(hours=w_second / sec['rate'])
        finish_B = max(fin_prim_split, fin_sec_split)
    
    final_mode = "Solo"
    if pd.notnull(finish_B) and finish_B <= finish_A: final_mode = "Split"
    
    total_idle_cost = timedelta(0)
    if final_mode == "Split":
        used_tipplers_list.append(sec['id'])
        idle_prim = max(timedelta(0), prim['eff_start'] - prim['free_at'])
        detailed_timings[f"{t_primary}_Idle"] = idle_prim
        detailed_timings[f"{t_primary}_Start"] = prim['eff_start']
        detailed_timings[f"{t_primary}_End"] = fin_prim_split
        total_idle_cost += idle_prim
        
        idle_sec = max(timedelta(0), real_start_sec - sec['free_at'])
        detailed_timings[f"{sec['id']}_Idle"] = idle_sec
        detailed_timings[f"{sec['id']}_Start"] = real_start_sec
        detailed_timings[f"{sec['id']}_End"] = fin_sec_split
        total_idle_cost += idle_sec
        
        actual_start_prim = prim['eff_start']
        overall_finish = finish_B
    else:
        idle_prim = max(timedelta(0), prim['eff_start'] - prim['free_at'])
        detailed_timings[f"{t_primary}_Idle"] = idle_prim
        detailed_timings[f"{t_primary}_Start"] = prim['eff_start']
        detailed_timings[f"{t_primary}_End"] = finish_A
        total_idle_cost += idle_prim
        actual_start_prim = prim['eff_start']
        overall_finish = finish_A

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

def run_full_simulation_initial(df_csv, params, df_locked, df_unplanned, start_seq_num):
    rates = {'T1': params['rt1'], 'T2': params['rt2'], 'T3': params['rt3'], 'T4': params['rt4']}
    s_a, s_b, extra_shunt_cross = params['sa'], params['sb'], params['extra_shunt']
    f_a, f_b, ft_hours = params['fa'], params['fb'], params['ft']
    w_batch, w_delay, downtimes = params['wb'], params['wd'], params['downtimes']

    to_plan = []
    
    # 1. Add Unplanned Actuals (from G-Sheet)
    if not df_unplanned.empty:
        for _, row in df_unplanned.iterrows():
            to_plan.append(row.to_dict())

    # 2. Add Forecast (CSV)
    if not df_csv.empty:
        df = df_csv.copy()
        load_col = find_column(df, ['LOAD TYPE', 'CMDT', 'COMMODITY'])
        if load_col: df = df[df[load_col].astype(str).str.strip().str.upper().str.startswith('BOXN')]
        
        wagon_col = find_column(df, ['TOTL UNTS', 'WAGONS', 'UNITS', 'TOTAL UNITS'])
        if wagon_col: df['wagon_count'] = df[wagon_col].apply(parse_wagons)
        else: df['wagon_count'] = 58 

        rake_col = find_column(df, ['RAKE NAME', 'RAKE', 'TRAIN NAME'])
        src_col = find_column(df, ['STTS FROM', 'STTN FROM', 'FROM_STN', 'SRC', 'SOURCE', 'FROM'])
        arvl_col = find_column(df, ['EXPD ARVLTIME', 'ARRIVAL TIME', 'EXPECTED ARRIVAL'])
        stts_time_col = find_column(df, ['STTS TIME'])
        stts_code_col = find_column(df, ['STTS CODE', 'STATUS'])

        if rake_col:
            df['exp_arrival_dt'] = pd.to_datetime(df[arvl_col], errors='coerce').apply(to_ist) if arvl_col else pd.NaT
            if stts_time_col and stts_code_col:
                df['stts_time_dt'] = pd.to_datetime(df[stts_time_col], errors='coerce').apply(to_ist)
                df['arrival_dt'] = df.apply(lambda r: r['stts_time_dt'] if str(r.get(stts_code_col)).strip()=='PL' and pd.notnull(r['stts_time_dt']) else r['exp_arrival_dt'], axis=1)
            else:
                df['arrival_dt'] = df['exp_arrival_dt']
            
            df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt')

            # DEDUPLICATION CHECK (By Arrival Time)
            existing_times = set()
            if not df_locked.empty:
                existing_times.update(df_locked['_Arrival_DT'].dt.floor('min'))
            if not df_unplanned.empty:
                existing_times.update(df_unplanned['_Arrival_DT'].dt.floor('min'))

            for _, row in df.iterrows():
                r_time = row['arrival_dt'].floor('min')
                if r_time in existing_times: continue 

                to_plan.append({
                    'Rake': row[rake_col],
                    'Coal Source': row.get(src_col, '') if src_col else '',
                    'Load Type': row.get(load_col, 'BOXN'),
                    'Wagons': row['wagon_count'],
                    'Status': row.get(stts_code_col, 'N/A') if stts_code_col else 'N/A',
                    '_Arrival_DT': row['arrival_dt'],
                    '_Form_Mins': 0,
                    'Optimization Type': 'Forecast',
                    'Extra Shunt (Mins)': 0,
                    'is_csv': True # Flag for renaming
                })

    plan_df = pd.DataFrame(to_plan)
    if plan_df.empty and df_locked.empty: return pd.DataFrame(), datetime.now(IST)
    
    if not plan_df.empty:
        plan_df = plan_df.sort_values('_Arrival_DT').reset_index(drop=True)
        first_arrival = plan_df['_Arrival_DT'].min()
    else:
        first_arrival = datetime.now(IST)

    sim_start_time = first_arrival.replace(hour=0, minute=0, second=0, microsecond=0)
    if sim_start_time.tzinfo is None: sim_start_time = IST.localize(sim_start_time)

    # REBUILD STATE from Locked Actuals
    tippler_state = {k: sim_start_time for k in ['T1', 'T2', 'T3', 'T4']}
    if not df_locked.empty:
        for _, row in df_locked.iterrows():
            used_str = str(row['_raw_tipplers'])
            end_val = row['_raw_end_dt']
            if pd.notnull(end_val):
                for t in ['T1', 'T2', 'T3', 'T4']:
                    if t in used_str and end_val > tippler_state[t]:
                        tippler_state[t] = end_val

    spec_line_status = {8: sim_start_time, 9: sim_start_time, 10: sim_start_time, 11: sim_start_time}
    line_groups = {
        'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}, 
        'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}
    }
    rr_tracker_A = -1 
    assignments = []
    
    current_seq = start_seq_num

    for _, rake in plan_df.iterrows():
        # RENAMING LOGIC
        display_name = rake['Rake']
        if rake.get('is_csv', False):
            current_seq += 1
            display_name = f"{current_seq}/{rake['Rake']}"

        options = []
        entry_A = get_line_entry_time('Group_Lines_8_10', rake['_Arrival_DT'], line_groups)
        ready_A = entry_A + timedelta(minutes=s_a)
        fin_A, used_A, start_A, tim_A, idle_A = calculate_generic_finish(
            rake['Wagons'], ['T1', 'T2'], ready_A, tippler_state, downtimes, rates, w_batch, w_delay)
        
        rel_A = fin_A + timedelta(minutes=f_a)
        dur_A = rel_A - rake['_Arrival_DT']
        dem_A_raw = max(0, (dur_A - timedelta(hours=ft_hours)).total_seconds())

        options.append({'id': 'Nat_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_A, 'start': start_A, 'fin': fin_A, 'used': used_A, 'timings': tim_A, 'idle': idle_A, 'dem_raw': dem_A_raw, 'extra_shunt': 0.0, 'form_mins': f_a, 'type': 'Standard'})

        entry_B = get_line_entry_time('Group_Line_11', rake['_Arrival_DT'], line_groups)
        ready_B = entry_B + timedelta(minutes=s_b)
        fin_B, used_B, start_B, tim_B, idle_B = calculate_generic_finish(
            rake['Wagons'], ['T3', 'T4'], ready_B, tippler_state, downtimes, rates, w_batch, w_delay)
        rel_B = fin_B + timedelta(minutes=f_b)
        dur_B = rel_B - rake['_Arrival_DT']
        dem_B_raw = max(0, (dur_B - timedelta(hours=ft_hours)).total_seconds())
        options.append({'id': 'Nat_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_B, 'start': start_B, 'fin': fin_B, 'used': used_B, 'timings': tim_B, 'idle': idle_B, 'dem_raw': dem_B_raw, 'extra_shunt': 0.0, 'form_mins': f_b, 'type': 'Standard'})

        ready_A_Cross = ready_A + timedelta(minutes=extra_shunt_cross)
        fin_AX, used_AX, start_AX, tim_AX, idle_AX = calculate_generic_finish(
            rake['Wagons'], ['T3', 'T4'], ready_A_Cross, tippler_state, downtimes, rates, w_batch, w_delay)
        rel_AX = fin_AX + timedelta(minutes=f_a)
        dur_AX = rel_AX - rake['_Arrival_DT']
        dem_AX_raw = max(0, (dur_AX - timedelta(hours=ft_hours)).total_seconds())
        options.append({'id': 'Cross_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_A_Cross, 'start': start_AX, 'fin': fin_AX, 'used': used_AX, 'timings': tim_AX, 'idle': idle_AX, 'dem_raw': dem_AX_raw, 'extra_shunt': float(extra_shunt_cross), 'form_mins': f_a, 'type': 'Cross-Transfer'})

        ready_B_Cross = ready_B + timedelta(minutes=extra_shunt_cross)
        fin_BX, used_BX, start_BX, tim_BX, idle_BX = calculate_generic_finish(
            rake['Wagons'], ['T1', 'T2'], ready_B_Cross, tippler_state, downtimes, rates, w_batch, w_delay)
        rel_BX = fin_BX + timedelta(minutes=f_b)
        dur_BX = rel_BX - rake['_Arrival_DT']
        dem_BX_raw = max(0, (dur_BX - timedelta(hours=ft_hours)).total_seconds())
        options.append({'id': 'Cross_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_B_Cross, 'start': start_BX, 'fin': fin_BX, 'used': used_BX, 'timings': tim_BX, 'idle': idle_BX, 'dem_raw': dem_BX_raw, 'extra_shunt': float(extra_shunt_cross), 'form_mins': f_b, 'type': 'Cross-Transfer'})

        best_opt = sorted(options, key=lambda x: (x['dem_raw'], x['fin'], x['idle']))[0]

        for k, v in best_opt['timings'].items():
            if 'End' in k: tippler_state[k.split('_')[0]] = v
        
        if best_opt['grp'] == 'Group_Lines_8_10':
            sel_line = find_specific_line(best_opt['grp'], best_opt['entry'], spec_line_status, rr_tracker_A)
            if sel_line in [8,9,10]: rr_tracker_A = {8:0,9:1,10:2}[sel_line]
        else: sel_line = 11
        
        line_groups[best_opt['grp']]['line_free_times'].append(best_opt['entry'] + timedelta(minutes=line_groups[best_opt['grp']]['clearance_mins']))
        spec_line_status[sel_line] = line_groups[best_opt['grp']]['line_free_times'][-1]

        release_time = best_opt['fin'] + timedelta(minutes=best_opt['form_mins'])
        tot_dur_final = release_time - rake['_Arrival_DT']
        dem_str, _ = calculate_rounded_demurrage(tot_dur_final, ft_hours)

        row_data = {
            'Rake': display_name,
            'Coal Source': rake['Coal Source'],
            'Load Type': rake['Load Type'],
            'Wagons': rake['Wagons'],
            'Status': rake['Status'],
            '_Arrival_DT': rake['_Arrival_DT'],
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
    
    if not df_locked.empty:
        actuals_clean = df_locked.drop(columns=['_raw_end_dt', '_raw_tipplers'], errors='ignore')
        final_df = pd.concat([actuals_clean, df_sim], ignore_index=True) if not df_sim.empty else actuals_clean
    else:
        final_df = df_sim

    return final_df, sim_start_time

def recalculate_cascade_reactive(edited_df, free_time_hours, sim_start_dt):
    recalc_rows = []
    if sim_start_dt.tzinfo is None: sim_start_dt = IST.localize(sim_start_dt)
    tippler_state = {k: sim_start_dt for k in ['T1', 'T2', 'T3', 'T4']}
    daily_demurrage_hours = {}
    
    # Pass 1: State
    for _, row in edited_df.iterrows():
        if row.get('Status') == 'ACTUAL':
            for t in ['T1', 'T2', 'T3', 'T4']:
                end_str = row.get(f"{t} End")
                if pd.notnull(end_str) and end_str != "":
                    end_dt = restore_dt(end_str, sim_start_dt)
                    if pd.notnull(end_dt) and end_dt > tippler_state[t]: tippler_state[t] = end_dt

    # Pass 2: Calc
    for _, row in edited_df.iterrows():
        if row.get('Status') == 'ACTUAL':
            dem_hrs = 0
            dem_val = str(row['Demurrage']).strip()
            if ":" in dem_val:
                try: dem_hrs = int(dem_val.split(":")[0])
                except: pass
            elif dem_val.isdigit(): dem_hrs = int(dem_val)
            elif dem_val in ["00:00", "", "nan"]:
                try:
                    dur_str = row['Total Duration']
                    sign = -1 if dur_str.startswith('-') else 1
                    parts = dur_str.replace('-','').split(':')
                    dur_td = timedelta(hours=int(parts[0]), minutes=int(parts[1])) * sign
                    _, dem_hrs = calculate_rounded_demurrage(dur_td, free_time_hours)
                except: pass
            
            try:
                arr_dt = pd.to_datetime(row['_Arrival_DT'])
                if arr_dt.tzinfo is None: arr_dt = IST.localize(arr_dt)
                d_str = arr_dt.strftime('%Y-%m-%d')
                daily_demurrage_hours[d_str] = daily_demurrage_hours.get(d_str, 0) + dem_hrs
            except: pass
            recalc_rows.append(row)
            continue
            
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

        release_time = max_finish + timedelta(minutes=float(row['_Form_Mins']))
        tot_dur = release_time - arrival
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
sim_params['ft'] = st.sidebar.number_input("Free Time (Hours)", value=7.0, step=0.5)
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

if gs_url and 'actuals_df_disp' not in st.session_state:
    st.session_state.actuals_df_disp = pd.DataFrame()
    
if gs_url:
    try:
        if 'actuals_df' not in st.session_state or st.session_state.actuals_df.empty:
             df_disp, _, _ = fetch_google_sheet_actuals(gs_url, sim_params['ft'])
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

uploaded_file = st.file_uploader("Upload FOIS CSV File (Plan)", type=["csv"])

input_changed = False
if uploaded_file and ('last_file_id' not in st.session_state or st.session_state.last_file_id != uploaded_file.file_id):
    input_changed = True
    st.session_state.last_file_id = uploaded_file.file_id
if gs_url and ('last_gs_url' not in st.session_state or st.session_state.last_gs_url != gs_url):
    input_changed = True
    st.session_state.last_gs_url = gs_url

if input_changed or 'raw_data_cached' not in st.session_state:
    actuals_df, unplanned_df, max_seq = pd.DataFrame(), pd.DataFrame(), 0
    if gs_url:
        actuals_df, unplanned_df, max_seq = fetch_google_sheet_actuals(gs_url, sim_params['ft'])
        st.session_state.actuals_df_disp = actuals_df
    
    st.session_state.actuals_df = actuals_df
    st.session_state.unplanned_df = unplanned_df
    st.session_state.max_seq = max_seq
    st.session_state.actuals_state = {}

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
    df_unplanned = st.session_state.get('unplanned_df', pd.DataFrame())
    start_seq = st.session_state.get('max_seq', 0)
    
    if input_changed or 'sim_result' not in st.session_state:
        sim_result, sim_start_dt = run_full_simulation_initial(df_raw, sim_params, df_act, df_unplanned, start_seq)
        st.session_state.sim_result = sim_result
        st.session_state.sim_start_dt = sim_start_dt

    if 'sim_result' in st.session_state and not st.session_state.sim_result.empty:
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
