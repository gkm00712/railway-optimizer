import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import pytz
import math
import numpy as np
import re

# ==========================================
# 1. PAGE CONFIGURATION & INPUTS
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer (IST)", layout="wide")
st.title("🚂 BOXN & BOBR Rake Logistics Dashboard (IST)")

IST = pytz.timezone('Asia/Kolkata')

# --- SIDEBAR INPUTS ---
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
# 2. HELPER FUNCTIONS
# ==========================================

def to_ist(dt):
    if pd.isnull(dt): return pd.NaT
    if dt.tzinfo is None: return IST.localize(dt)
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
        if cand_upper in cols_upper: return df.columns[cols_upper.index(cand_upper)]
        for c in cols_upper:
            if cand_upper == c: return df.columns[cols_upper.index(c)]
    return None

def parse_last_sequence(rake_name):
    try:
        s = str(rake_name).strip()
        match_complex = re.search(r'(\d+)\D+(\d+)', s)
        if match_complex: return int(match_complex.group(1)), int(match_complex.group(2))
        match_single = re.search(r'^(\d+)', s)
        if match_single:
            val = int(match_single.group(1))
            if val > 1000: return 0, val
            return val, 0
    except: pass
    return 0, 0

def parse_tippler_cell(cell_value, ref_date):
    if pd.isnull(cell_value): return pd.NaT, pd.NaT
    s = str(cell_value).strip()
    times_found = re.findall(r'(\d{1,2}:\d{2})', s)
    if len(times_found) >= 2:
        start_str, end_str = times_found[0], times_found[1]
        try:
            if ref_date.tzinfo is None: ref_date = IST.localize(ref_date)
            s_h, s_m = map(int, start_str.split(':'))
            e_h, e_m = map(int, end_str.split(':'))
            start_dt = ref_date.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
            end_dt = ref_date.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
            if end_dt < start_dt: end_dt += timedelta(days=1)
            if (start_dt - ref_date).total_seconds() < -43200: 
                 start_dt += timedelta(days=1); end_dt += timedelta(days=1)
            return start_dt, end_dt
        except: pass
    return pd.NaT, pd.NaT

def parse_col_d_wagon_type(cell_val):
    wagons = 58 
    load_type = 'BOXN' 
    if pd.isnull(cell_val): return wagons, load_type
    s = str(cell_val).strip().upper()
    match_num = re.search(r'(\d{2})', s)
    if match_num:
        try: wagons = int(match_num.group(1))
        except: pass
    if 'R' in s: load_type = 'BOBR'
    elif 'N' in s: load_type = 'BOXN'
    return wagons, load_type

# ==========================================
# 3. GOOGLE SHEET PARSER
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

        locked_actuals = []
        unplanned_actuals = [] 
        
        today_date = datetime.now(IST).date()
        last_seq_tuple = (0, 0)

        for _, row in df_gs.iterrows():
            # --- 1. VALIDITY CHECK (B & C Must be Filled) ---
            # B=Index 1, C=Index 2
            val_b = str(row.iloc[1]).strip()
            val_c = str(row.iloc[2]).strip()
            if not val_b or not val_c or val_b.lower() == 'nan' or val_c.lower() == 'nan':
                continue # Skip row completely

            arrival_dt = safe_parse_date(row.iloc[4]) 
            if pd.isnull(arrival_dt): continue
            
            # --- 2. SEQUENCE TRACKING ---
            rake_name = val_b
            seq, rid = parse_last_sequence(rake_name)
            if seq > last_seq_tuple[0] or (seq == last_seq_tuple[0] and rid > last_seq_tuple[1]):
                last_seq_tuple = (seq, rid)

            source_val = val_c
            
            # --- 3. PARSE COL D (Index 3) ---
            col_d_val = row.iloc[3]
            wagons, load_type = parse_col_d_wagon_type(col_d_val)

            start_dt = safe_parse_date(row.iloc[5])
            end_dt = safe_parse_date(row.iloc[6])
            if pd.isnull(start_dt): start_dt = arrival_dt 
            
            # --- 4. TIPPLER PARSING & CHECK ---
            tippler_timings = {}
            used_tipplers = []
            
            for t_name, idx in [('T1', 14), ('T2', 15), ('T3', 16), ('T4', 17)]:
                cell_val = row.iloc[idx]
                if pd.notnull(cell_val) and str(cell_val).strip() not in ["", "nan"]:
                    ts, te = parse_tippler_cell(cell_val, arrival_dt)
                    if pd.isnull(ts) and pd.notnull(start_dt):
                        ts = start_dt
                        te = end_dt if pd.notnull(end_dt) else start_dt + timedelta(hours=2)
                    
                    if pd.notnull(ts):
                        used_tipplers.append(t_name)
                        tippler_timings[f"{t_name} Start"] = format_dt(ts)
                        tippler_timings[f"{t_name} End"] = format_dt(te)
                        tippler_timings[f"{t_name}_Obj_End"] = te

            # --- 5. LOGIC SPLIT ---
            # Case A: Unplanned (No Tipplers Assigned)
            if not used_tipplers and load_type != 'BOBR':
                unplanned_actuals.append({
                    'Coal Source': source_val,
                    'Load Type': load_type,
                    'Wagons': wagons,
                    'Status': 'Pending (G-Sheet)',
                    '_Arrival_DT': arrival_dt,
                    '_Form_Mins': 0,
                    'Optimization Type': 'Auto-Planned (G-Sheet)',
                    'Extra Shunt (Mins)': 0,
                    'is_gs_unplanned': True 
                })
                # Note: We do NOT filter unplanned by date (backlog must be cleared)
                continue 

            # Case B: Locked Actuals
            # Filter Logic: "dont show yesterday data even if unloading ends on current day"
            # Strict Interpretation: If Arrival < Today, HIDE IT.
            if arrival_dt.date() < today_date:
                continue

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

            dem_str = "00:00"
            raw_dem = row.iloc[10] 
            if pd.notnull(raw_dem) and str(raw_dem).strip() != "":
                clean = str(raw_dem).strip()
                if ":" in clean: dem_str = clean
                elif clean.replace('.','',1).isdigit(): dem_str = f"{int(float(clean)):02d}:00"

            entry = {
                'Rake': rake_name,
                'Coal Source': source_val,
                'Load Type': load_type,
                'Wagons': wagons,
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
            
            locked_actuals.append(entry)

        return pd.DataFrame(locked_actuals), pd.DataFrame(unplanned_actuals), last_seq_tuple

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
        predicted_finish = effective_start + timedelta(hours=wagons / rates[t])
        candidates.append({'id': t, 'rate': rates[t], 'eff_start': effective_start, 'free_at': free_at, 'pred_fin': predicted_finish})
    
    sorted_tipplers = sorted(candidates, key=lambda x: x['pred_fin'])
    prim = sorted_tipplers[0]
    t_primary = prim['id']
    
    detailed_timings = {}
    used_tipplers_list = [t_primary]
    
    finish_A = prim['pred_fin']
    
    idle_prim = max(timedelta(0), prim['eff_start'] - prim['free_at'])
    detailed_timings[f"{t_primary}_Idle"] = idle_prim
    detailed_timings[f"{t_primary}_Start"] = prim['eff_start']
    detailed_timings[f"{t_primary}_End"] = finish_A
    
    return finish_A, ", ".join(used_tipplers_list), prim['eff_start'], detailed_timings, idle_prim

def get_line_entry_time(group, arrival, line_groups):
    grp = line_groups[group]
    active = sorted([t for t in grp['line_free_times'] if t > arrival])
    if len(active) < grp['capacity']: return arrival
    return active[len(active) - grp['capacity']]

def run_full_simulation_initial(df_csv, params, df_locked, df_unplanned, last_seq_tuple):
    rates = {'T1': params['rt1'], 'T2': params['rt2'], 'T3': params['rt3'], 'T4': params['rt4']}
    s_a, s_b, extra_shunt_cross = params['sa'], params['sb'], params['extra_shunt']
    f_a, f_b, ft_hours = params['fa'], params['fb'], params['ft']
    w_batch, w_delay, downtimes = params['wb'], params['wd'], params['downtimes']

    to_plan = []
    
    if not df_unplanned.empty:
        for _, row in df_unplanned.iterrows():
            to_plan.append(row.to_dict())

    if not df_csv.empty:
        df = df_csv.copy()
        load_col = find_column(df, ['LOAD TYPE', 'CMDT', 'COMMODITY'])
        if load_col:
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
            if not df_locked.empty: existing_times.update(df_locked['_Arrival_DT'].dt.floor('min'))
            if not df_unplanned.empty: existing_times.update(df_unplanned['_Arrival_DT'].dt.floor('min'))

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

    # State from Locked
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

    line_groups = {
        'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []},
        'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}
    }
    
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
        fin_A, used_A, start_A, tim_A, _ = calculate_generic_finish(
            rake['Wagons'], ['T1', 'T2'], ready_A, tippler_state, downtimes, rates, w_batch, w_delay)
        
        entry_B = get_line_entry_time('Group_Line_11', rake['_Arrival_DT'], line_groups)
        ready_B = entry_B + timedelta(minutes=s_b)
        fin_B, used_B, start_B, tim_B, _ = calculate_generic_finish(
            rake['Wagons'], ['T3', 'T4'], ready_B, tippler_state, downtimes, rates, w_batch, w_delay)
        
        if fin_B < fin_A:
            best_fin, best_used, best_start, best_timings, best_entry, best_ready = fin_B, used_B, start_B, tim_B, entry_B, ready_B
            best_grp, best_line = 'Group_Line_11', '11'
            best_type = "Standard (Fast)"
        else:
            best_fin, best_used, best_start, best_timings, best_entry, best_ready = fin_A, used_A, start_A, tim_A, entry_A, ready_A
            best_grp, best_line = 'Group_Lines_8_10', '8/9/10'
            best_type = "Standard"

        for k, v in best_timings.items():
            if 'End' in k: tippler_state[k.split('_')[0]] = v
        if best_grp == 'Group_Lines_8_10':
             line_groups['Group_Lines_8_10']['line_free_times'].append(best_entry + timedelta(minutes=50))

        release_time = best_fin + timedelta(minutes=f_a)
        tot_dur_final = release_time - rake['_Arrival_DT']
        dem_str, _ = calculate_rounded_demurrage(tot_dur_final, ft_hours)

        row_data = {
            'Rake': display_name,
            'Coal Source': rake['Coal Source'],
            'Load Type': rake['Load Type'],
            'Wagons': rake['Wagons'],
            'Status': rake['Status'],
            '_Arrival_DT': rake['_Arrival_DT'],
            '_Shunt_Ready_DT': best_ready,
            '_Form_Mins': f_a,
            'Optimization Type': best_type,
            'Extra Shunt (Mins)': 0,
            'Line Allotted': best_line,
            'Line Entry Time': format_dt(best_entry),
            'Shunting Complete': format_dt(best_ready),
            'Tippler Start Time': format_dt(best_start),
            'Finish Unload': format_dt(best_fin),
            'Tipplers Used': best_used,
            'Wait (Tippler)': format_duration_hhmm(best_start - best_ready),
            'Total Duration': format_duration_hhmm(tot_dur_final),
            'Demurrage': dem_str
        }
        for t in ['T1', 'T2', 'T3', 'T4']:
             row_data[f"{t} Start"] = format_dt(best_timings.get(f"{t}_Start", pd.NaT))
             row_data[f"{t} End"] = format_dt(best_timings.get(f"{t}_End", pd.NaT))
             row_data[f"{t} Idle"] = format_duration_hhmm(best_timings.get(f"{t}_Idle", pd.NaT))
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

uploaded_file = st.file_uploader("Upload FOIS CSV File (Plan)", type=["csv"])

input_changed = False
if uploaded_file and ('last_file_id' not in st.session_state or st.session_state.last_file_id != uploaded_file.file_id):
    input_changed = True
    st.session_state.last_file_id = uploaded_file.file_id
if gs_url and ('last_gs_url' not in st.session_state or st.session_state.last_gs_url != gs_url):
    input_changed = True
    st.session_state.last_gs_url = gs_url

if input_changed or 'raw_data_cached' not in st.session_state:
    actuals_df, unplanned_df, last_seq = pd.DataFrame(), pd.DataFrame(), (0,0)
    if gs_url:
        actuals_df, unplanned_df, last_seq = fetch_google_sheet_actuals(gs_url, sim_params['ft'])
    st.session_state.actuals_df = actuals_df
    st.session_state.unplanned_df = unplanned_df
    st.session_state.last_seq = last_seq
    if uploaded_file:
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file)
            df_raw.columns = df_raw.columns.str.strip().str.upper()
            st.session_state.raw_data_cached = df_raw
        except Exception: st.stop()
    else: st.session_state.raw_data_cached = pd.DataFrame()

if 'raw_data_cached' in st.session_state or 'actuals_df' in st.session_state:
    df_raw = st.session_state.get('raw_data_cached', pd.DataFrame())
    df_act = st.session_state.get('actuals_df', pd.DataFrame())
    df_unplanned = st.session_state.get('unplanned_df', pd.DataFrame())
    start_seq = st.session_state.get('last_seq', (0,0))
    
    if input_changed or 'sim_result' not in st.session_state:
        sim_result, sim_start_dt = run_full_simulation_initial(df_raw, sim_params, df_act, df_unplanned, start_seq)
        st.session_state.sim_result = sim_result
        st.session_state.sim_start_dt = sim_start_dt

    if 'sim_result' in st.session_state and not st.session_state.sim_result.empty:
        df_final = st.session_state.sim_result
        df_final['Date_Str'] = df_final['_Arrival_DT'].dt.strftime('%Y-%m-%d')
        unique_dates = sorted(df_final['Date_Str'].unique())

        col_cfg = {
            "Rake": st.column_config.TextColumn("Rake Name", disabled=True),
            "Coal Source": st.column_config.TextColumn("Source/Mine", disabled=True),
            "Status": st.column_config.TextColumn("Status", help="ACTUAL = From G-Sheet"),
            "Tippler Start Time": st.column_config.TextColumn("Start (dd-HH:MM)"),
            "Finish Unload": st.column_config.TextColumn("Finish (dd-HH:MM)"),
            "Extra Shunt (Mins)": st.column_config.NumberColumn("Ext. Shunt", step=5),
            "_Arrival_DT": None, "_Shunt_Ready_DT": None, "_Form_Mins": None, "Date_Str": None
        }

        for d in unique_dates:
            st.markdown(f"### 📅 Schedule for {d}")
            day_df = df_final[df_final['Date_Str'] == d].copy()
            day_df.index = np.arange(1, len(day_df) + 1)
            
            st.data_editor(
                day_df,
                use_container_width=True,
                num_rows="fixed",
                column_config=col_cfg,
                disabled=["Rake", "Coal Source", "Load Type", "Wagons", "Line Allotted", "Wait (Tippler)", "Total Duration", "Demurrage", "Tipplers Used", "Status", "T1 Start", "T1 End", "T2 Start", "T2 End", "T3 Start", "T3 End", "T4 Start", "T4 End"],
                key=f"editor_{d}"
            )

        _, daily_stats = recalculate_cascade_reactive(df_final, sim_params['ft'], st.session_state.sim_start_dt)
        st.markdown("### 📅 Demurrage Forecast")
        st.dataframe(pd.DataFrame(list(daily_stats.items()), columns=['Date', 'Total Hours']).assign(Demurrage=lambda x: x['Total Hours'].apply(lambda h: f"{int(h)} Hours"))[['Date', 'Demurrage']], hide_index=True)
        
        st.download_button("📥 Download Final Report", df_final.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins", "Date_Str"]).to_csv(index=False).encode('utf-8'), "optimized_schedule.csv", "text/csv")
