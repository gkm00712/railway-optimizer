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
st.title("🚂 BOXN & BOBR Rake Logistics Dashboard (IST)")

# Define IST Timezone
IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def to_ist(dt):
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
    if pd.isnull(duration_delta): return "00:00", 0
    total_minutes = duration_delta.total_seconds() / 60
    free_minutes = free_time_hours * 60
    demurrage_minutes = total_minutes - free_minutes
    if demurrage_minutes <= 0: return "00:00", 0
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
            if cand_upper == c: return df.columns[cols_upper.index(c)]
    return None

def parse_last_sequence(rake_name):
    try:
        s = str(rake_name).strip()
        match_complex = re.search(r'(\d+)\D+(\d+)', s)
        if match_complex:
            return int(match_complex.group(1)), int(match_complex.group(2))
        match_single = re.search(r'^(\d+)', s)
        if match_single:
            val = int(match_single.group(1))
            if val > 1000: return 0, val
            return val, 0
    except: pass
    return 0, 0

def parse_tippler_cell(cell_value, ref_date):
    """
    Parses '13(06:40-08:45)' -> Returns (Start_DT, End_DT)
    """
    if pd.isnull(cell_value): return pd.NaT, pd.NaT
    s = str(cell_value).strip()
    match = re.search(r'\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)', s)
    if match:
        start_str, end_str = match.group(1), match.group(2)
        try:
            if ref_date.tzinfo is None: ref_date = IST.localize(ref_date)
            
            s_h, s_m = map(int, start_str.split(':'))
            e_h, e_m = map(int, end_str.split(':'))
            
            start_dt = ref_date.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
            end_dt = ref_date.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
            
            if end_dt < start_dt:
                end_dt += timedelta(days=1)
            
            # Robustness: If start time is way earlier than ref date (e.g. prev day), adjust
            if (start_dt - ref_date).total_seconds() < -43200: # 12 hours ago
                 start_dt += timedelta(days=1)
                 end_dt += timedelta(days=1)

            return start_dt, end_dt
        except: pass
    return pd.NaT, pd.NaT

# ==========================================
# 3. GOOGLE SHEET PARSER (Cached)
# ==========================================

def safe_parse_date(val):
    if pd.isnull(val) or str(val).strip() == "" or str(val).strip().upper() == "U/P": return pd.NaT
    try:
        dt = pd.to_datetime(val, dayfirst=True, errors='coerce') 
        if pd.isnull(dt): return pd.NaT 
        return to_ist(dt)
    except: return pd.NaT

@st.cache_data(ttl=60)
def fetch_google_sheet_actuals(url, free_time_hours):
    try:
        df_gs = pd.read_csv(url, header=None, skiprows=1) 
        if len(df_gs.columns) < 18: return pd.DataFrame(), pd.DataFrame(), (0,0)

        actuals = []
        today_date = datetime.now(IST).date()
        last_seq_tuple = (0, 0)

        for _, row in df_gs.iterrows():
            arrival_dt = safe_parse_date(row.iloc[4]) 
            if pd.isnull(arrival_dt): continue
            
            rake_name = str(row.iloc[1])
            seq, rid = parse_last_sequence(rake_name)
            if seq > last_seq_tuple[0] or (seq == last_seq_tuple[0] and rid > last_seq_tuple[1]):
                last_seq_tuple = (seq, rid)

            if arrival_dt.date() != today_date: continue

            source_val = str(row.iloc[2]) 
            if source_val.lower() == 'nan': source_val = ""
            
            # Load Type Logic: Check for BOBR in name if column is missing, or trust logic later
            # Usually Load Type is not in G-Sheet explicit columns, assume BOXN unless name says otherwise
            load_type = 'BOXN'
            if 'BOBR' in str(row.iloc[1]).upper(): load_type = 'BOBR'

            start_dt = safe_parse_date(row.iloc[5])
            end_dt = safe_parse_date(row.iloc[6])
            
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

            if pd.isnull(start_dt): start_dt = arrival_dt 

            dem_str = "00:00"
            raw_dem = row.iloc[10] 
            if pd.notnull(raw_dem) and str(raw_dem).strip() != "":
                clean = str(raw_dem).strip()
                if ":" in clean: dem_str = clean
                elif clean.replace('.','',1).isdigit(): dem_str = f"{int(float(clean)):02d}:00"
            
            tippler_timings = {}
            used_tipplers = []
            
            # Parse O, P, Q, R
            for t_name, idx in [('T1', 14), ('T2', 15), ('T3', 16), ('T4', 17)]:
                cell_val = row.iloc[idx]
                if pd.notnull(cell_val) and str(cell_val).strip() not in ["", "nan"]:
                    ts, te = parse_tippler_cell(cell_val, arrival_dt)
                    if pd.notnull(ts):
                        used_tipplers.append(t_name)
                        tippler_timings[f"{t_name} Start"] = format_dt(ts)
                        tippler_timings[f"{t_name} End"] = format_dt(te)
                        tippler_timings[f"{t_name}_Obj_End"] = te

            entry = {
                'Rake': rake_name,
                'Coal Source': source_val,
                'Load Type': load_type,
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
                'Tipplers Used': ", ".join(used_tipplers),
                'Wait (Tippler)': format_duration_hhmm(start_dt - arrival_dt) if pd.notnull(start_dt) else "",
                'Total Duration': format_duration_hhmm(total_dur),
                'Demurrage': dem_str,
                '_raw_end_dt': end_dt,
                '_raw_tipplers_data': tippler_timings,
                '_raw_tipplers': used_tipplers
            }
            
            for t in ['T1', 'T2', 'T3', 'T4']:
                entry[f"{t} Start"] = tippler_timings.get(f"{t} Start", "")
                entry[f"{t} End"] = tippler_timings.get(f"{t} End", "")
                entry[f"{t} Idle"] = ""
            
            actuals.append(entry)

        return pd.DataFrame(actuals), pd.DataFrame(), last_seq_tuple

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), (0,0)

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

def run_full_simulation_initial(df_csv, params, df_locked, last_seq_tuple):
    rates = {'T1': params['rt1'], 'T2': params['rt2'], 'T3': params['rt3'], 'T4': params['rt4']}
    s_a, s_b, extra_shunt_cross = params['sa'], params['sb'], params['extra_shunt']
    f_a, f_b, ft_hours = params['fa'], params['fb'], params['ft']
    w_batch, w_delay, downtimes = params['wb'], params['wd'], params['downtimes']

    to_plan = []
    
    if not df_csv.empty:
        df = df_csv.copy()
        
        # 1. ALLOW BOXN & BOBR (RELAXED MATCH)
        load_col = find_column(df, ['LOAD TYPE', 'CMDT', 'COMMODITY'])
        if load_col:
            # Filter: Check if string contains BOXN OR BOBR (case insensitive)
            df = df[df[load_col].astype(str).str.upper().str.contains('BOXN|BOBR', regex=True, na=False)]
        
        wagon_col = find_column(df, ['TOTL UNTS', 'WAGONS', 'UNITS', 'TOTAL UNITS'])
        if wagon_col: df['wagon_count'] = df[wagon_col].apply(parse_wagons)
        else: df['wagon_count'] = 58 

        src_col = find_column(df, ['STTS FROM', 'STTN FROM', 'FROM_STN', 'SRC', 'SOURCE', 'FROM'])
        arvl_col = find_column(df, ['EXPD ARVLTIME', 'ARRIVAL TIME', 'EXPECTED ARRIVAL'])
        stts_time_col = find_column(df, ['STTS TIME'])
        stts_code_col = find_column(df, ['STTS CODE', 'STATUS'])

        if arvl_col:
            df['exp_arrival_dt'] = pd.to_datetime(df[arvl_col], errors='coerce').apply(to_ist) if arvl_col else pd.NaT
            if stts_time_col and stts_code_col:
                df['stts_time_dt'] = pd.to_datetime(df[stts_time_col], errors='coerce').apply(to_ist)
                df['arrival_dt'] = df.apply(lambda r: r['stts_time_dt'] if str(r.get(stts_code_col)).strip()=='PL' and pd.notnull(r['stts_time_dt']) else r['exp_arrival_dt'], axis=1)
            else:
                df['arrival_dt'] = df['exp_arrival_dt']
            
            df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt')

            existing_times = set()
            if not df_locked.empty:
                existing_times.update(df_locked['_Arrival_DT'].dt.floor('min'))

            for _, row in df.iterrows():
                if row['arrival_dt'].floor('min') in existing_times: continue 
                
                l_type = row.get(load_col, 'BOXN') if load_col else 'BOXN'

                to_plan.append({
                    'Coal Source': row.get(src_col, '') if src_col else '',
                    'Load Type': l_type,
                    'Wagons': row['wagon_count'],
                    'Status': row.get(stts_code_col, 'N/A') if stts_code_col else 'N/A',
                    '_Arrival_DT': row['arrival_dt'],
                    '_Form_Mins': 0,
                    'Optimization Type': 'Forecast',
                    'Extra Shunt (Mins)': 0,
                    'is_csv': True 
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

    tippler_state = {k: sim_start_time for k in ['T1', 'T2', 'T3', 'T4']}
    if not df_locked.empty:
        for _, row in df_locked.iterrows():
            raw_t_data = row.get('_raw_tipplers_data', {})
            parsed_found = False
            for t in ['T1', 'T2', 'T3', 'T4']:
                t_end_obj = raw_t_data.get(f"{t}_Obj_End")
                if pd.notnull(t_end_obj):
                    if t_end_obj > tippler_state[t]: tippler_state[t] = t_end_obj
                    parsed_found = True
            
            if not parsed_found:
                used_str = str(row['_raw_tipplers'])
                end_val = row['_raw_end_dt']
                if pd.notnull(end_val):
                    for t in ['T1', 'T2', 'T3', 'T4']:
                        if t in used_str and end_val > tippler_state[t]:
                            tippler_state[t] = end_val

    line_groups = {'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}}
    curr_seq = last_seq_tuple[0]
    curr_id = last_seq_tuple[1]
    assignments = []

    for _, rake in plan_df.iterrows():
        curr_seq += 1
        curr_id += 1
        display_name = f"{curr_seq}/{curr_id}"
        
        is_bobr = 'BOBR' in str(rake['Load Type']).upper()

        if is_bobr:
            row_data = {
                'Rake': display_name,
                'Coal Source': rake['Coal Source'],
                'Load Type': rake['Load Type'],
                'Wagons': rake['Wagons'],
                'Status': rake['Status'],
                '_Arrival_DT': rake['_Arrival_DT'],
                '_Shunt_Ready_DT': pd.NaT,
                '_Form_Mins': 0,
                'Optimization Type': 'BOBR (No Tippler)',
                'Extra Shunt (Mins)': 0,
                'Line Allotted': 'N/A',
                'Line Entry Time': format_dt(rake['_Arrival_DT']),
                'Shunting Complete': "",
                'Tippler Start Time': "",
                'Finish Unload': "",
                'Tipplers Used': "N/A",
                'Wait (Tippler)': "",
                'Total Duration': "",
                'Demurrage': "00:00"
            }
            for t in ['T1', 'T2', 'T3', 'T4']:
                 row_data[f"{t} Start"] = ""; row_data[f"{t} End"] = ""; row_data[f"{t} Idle"] = ""
            assignments.append(row_data)
            continue

        entry_A = get_line_entry_time('Group_Lines_8_10', rake['_Arrival_DT'], line_groups)
        ready_A = entry_A + timedelta(minutes=s_a)
        fin_A, used_A, start_A, tim_A, idle_A = calculate_generic_finish(
            rake['Wagons'], ['T1', 'T2'], ready_A, tippler_state, downtimes, rates, w_batch, w_delay)
        
        for k, v in tim_A.items():
            if 'End' in k: tippler_state[k.split('_')[0]] = v
        
        line_groups['Group_Lines_8_10']['line_free_times'].append(entry_A + timedelta(minutes=50))

        release_time = fin_A + timedelta(minutes=f_a)
        tot_dur_final = release_time - rake['_Arrival_DT']
        dem_str, _ = calculate_rounded_demurrage(tot_dur_final, ft_hours)

        row_data = {
            'Rake': display_name,
            'Coal Source': rake['Coal Source'],
            'Load Type': rake['Load Type'],
            'Wagons': rake['Wagons'],
            'Status': rake['Status'],
            '_Arrival_DT': rake['_Arrival_DT'],
            '_Shunt_Ready_DT': ready_A,
            '_Form_Mins': f_a,
            'Optimization Type': 'Forecast',
            'Extra Shunt (Mins)': 0,
            'Line Allotted': '8/9/10',
            'Line Entry Time': format_dt(entry_A),
            'Shunting Complete': format_dt(ready_A),
            'Tippler Start Time': format_dt(start_A),
            'Finish Unload': format_dt(fin_A),
            'Tipplers Used': used_A,
            'Wait (Tippler)': format_duration_hhmm(start_A - ready_A),
            'Total Duration': format_duration_hhmm(tot_dur_final),
            'Demurrage': dem_str
        }
        for t in ['T1', 'T2', 'T3', 'T4']:
             row_data[f"{t} Start"] = format_dt(tim_A.get(f"{t}_Start", pd.NaT))
             row_data[f"{t} End"] = format_dt(tim_A.get(f"{t}_End", pd.NaT))
             row_data[f"{t} Idle"] = format_duration_hhmm(tim_A.get(f"{t}_Idle", pd.NaT))
        assignments.append(row_data)

    df_sim = pd.DataFrame(assignments)
    
    if not df_locked.empty:
        cols_to_drop = ['_raw_tipplers_data', '_raw_end_dt', '_raw_tipplers']
        actuals_clean = df_locked.drop(columns=[c for c in cols_to_drop if c in df_locked.columns], errors='ignore')
        final_df = pd.concat([actuals_clean, df_sim], ignore_index=True) if not df_sim.empty else actuals_clean
    else:
        final_df = df_sim

    return final_df, sim_start_time

def recalculate_cascade_reactive(edited_df, free_time_hours, sim_start_dt):
    recalc_rows = []
    daily_demurrage_hours = {}
    
    for _, row in edited_df.iterrows():
        try:
            dem_val = str(row['Demurrage']).strip()
            dem_hrs = 0
            if ":" in dem_val: dem_hrs = int(dem_val.split(":")[0])
            elif dem_val.isdigit(): dem_hrs = int(dem_val)
            
            arr_dt = pd.to_datetime(row['_Arrival_DT'])
            if arr_dt.tzinfo is None: arr_dt = IST.localize(arr_dt)
            d_str = arr_dt.strftime('%Y-%m-%d')
            daily_demurrage_hours[d_str] = daily_demurrage_hours.get(d_str, 0) + dem_hrs
        except: pass
        recalc_rows.append(row)
        
    return pd.DataFrame(recalc_rows), daily_demurrage_hours

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if gs_url:
    try:
        if 'actuals_df' not in st.session_state or st.session_state.actuals_df.empty:
             _, _, _ = fetch_google_sheet_actuals(gs_url, sim_params['ft'])
    except: pass

uploaded_file = st.file_uploader("Upload FOIS CSV File (Plan)", type=["csv"])

input_changed = False
if uploaded_file and ('last_file_id' not in st.session_state or st.session_state.last_file_id != uploaded_file.file_id):
    input_changed = True
    st.session_state.last_file_id = uploaded_file.file_id
if gs_url and ('last_gs_url' not in st.session_state or st.session_state.last_gs_url != gs_url):
    input_changed = True
    st.session_state.last_gs_url = gs_url

if input_changed or 'raw_data_cached' not in st.session_state:
    actuals_df, _, last_seq = pd.DataFrame(), pd.DataFrame(), (0,0)
    if gs_url:
        actuals_df, _, last_seq = fetch_google_sheet_actuals(gs_url, sim_params['ft'])
    
    st.session_state.actuals_df = actuals_df
    st.session_state.last_seq = last_seq

    if uploaded_file:
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file)
            df_raw.columns = df_raw.columns.str.strip().str.upper()
            st.session_state.raw_data_cached = df_raw
        except Exception as e: st.stop()
    else: st.session_state.raw_data_cached = pd.DataFrame()

if 'raw_data_cached' in st.session_state or 'actuals_df' in st.session_state:
    df_raw = st.session_state.get('raw_data_cached', pd.DataFrame())
    df_act = st.session_state.get('actuals_df', pd.DataFrame())
    start_seq = st.session_state.get('last_seq', (0,0))
    
    if input_changed or 'sim_result' not in st.session_state:
        sim_result, sim_start_dt = run_full_simulation_initial(df_raw, sim_params, df_act, start_seq)
        st.session_state.sim_result = sim_result
        st.session_state.sim_start_dt = sim_start_dt

    if 'sim_result' in st.session_state and not st.session_state.sim_result.empty:
        # Group by Date
        df_final = st.session_state.sim_result
        df_final['Date'] = df_final['_Arrival_DT'].dt.date
        unique_dates = sorted(df_final['Date'].unique())

        col_cfg = {
            "Rake": st.column_config.TextColumn("Rake Name", disabled=True),
            "Coal Source": st.column_config.TextColumn("Source/Mine", disabled=True),
            "Status": st.column_config.TextColumn("Status", help="ACTUAL = From G-Sheet"),
            "Tippler Start Time": st.column_config.TextColumn("Start (dd-HH:MM)"),
            "Finish Unload": st.column_config.TextColumn("Finish (dd-HH:MM)"),
            "Extra Shunt (Mins)": st.column_config.NumberColumn("Ext. Shunt", step=5),
            "_Arrival_DT": None, "_Shunt_Ready_DT": None, "_Form_Mins": None, "Date": None
        }

        for d in unique_dates:
            st.markdown(f"### 📅 Schedule for {d.strftime('%d-%b-%Y')}")
            day_df = df_final[df_final['Date'] == d].copy()
            # 1-Based Indexing
            day_df.index = np.arange(1, len(day_df) + 1)
            
            st.data_editor(
                day_df,
                use_container_width=True,
                num_rows="fixed",
                column_config=col_cfg,
                disabled=["Rake", "Coal Source", "Load Type", "Wagons", "Line Allotted", "Wait (Tippler)", "Total Duration", "Demurrage", "Tipplers Used", "Status"],
                key=f"editor_{d}"
            )

        _, daily_stats = recalculate_cascade_reactive(df_final, sim_params['ft'], st.session_state.sim_start_dt)
        st.markdown("### 📅 Demurrage Forecast")
        st.dataframe(pd.DataFrame(list(daily_stats.items()), columns=['Date', 'Total Hours']).assign(Demurrage=lambda x: x['Total Hours'].apply(lambda h: f"{int(h)} Hours"))[['Date', 'Demurrage']], hide_index=True)
        
        st.download_button("📥 Download Final Report", df_final.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins", "Date"]).to_csv(index=False).encode('utf-8'), "optimized_schedule.csv", "text/csv")
