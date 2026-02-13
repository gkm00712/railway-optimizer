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

show_history = st.sidebar.checkbox("Show All Historical Data", value=False, help="Check this to see old completed rakes")

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

curr_params_hash = str(sim_params)
params_changed = False
if 'last_params_hash' not in st.session_state:
    st.session_state.last_params_hash = curr_params_hash
elif st.session_state.last_params_hash != curr_params_hash:
    params_changed = True
    st.session_state.last_params_hash = curr_params_hash

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

# FIXED: Logic to use Reference Date (Arrival) instead of Current Date
def parse_dt_from_str_smart(dt_str, ref_dt_obj):
    try:
        if not dt_str: return pd.NaT
        parts = dt_str.split('-')
        day = int(parts[0])
        hm = parts[1].split(':')
        h, m = int(hm[0]), int(hm[1])
        
        # Use Month/Year from Reference Object (Arrival Date)
        if pd.isnull(ref_dt_obj):
            year_ref = datetime.now().year
            month_ref = datetime.now().month
        else:
            year_ref = ref_dt_obj.year
            month_ref = ref_dt_obj.month
            
        dt = datetime(year_ref, month_ref, day, h, m)
        
        # Handle month rollover (e.g. Arrival 31st Jan, Unload 1st Feb)
        # If created date is much earlier than arrival, it likely belongs to next month
        if pd.notnull(ref_dt_obj):
             naive_ref = ref_dt_obj.replace(tzinfo=None)
             if (dt - naive_ref).days < -20: # If it looks like a month behind
                 # Add roughly a month? Simplest is to rely on the day number
                 if month_ref == 12:
                     dt = dt.replace(year=year_ref+1, month=1)
                 else:
                     dt = dt.replace(month=month_ref+1)
                     
        return IST.localize(dt)
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

def parse_wagon_count_from_cell(cell_value):
    if pd.isnull(cell_value): return None
    s = str(cell_value).strip()
    clean_s = re.sub(r'\d{1,2}:\d{2}', '', s)
    matches = re.findall(r'\b(\d{1,3})\b', clean_s)
    if matches:
        return int(matches[0]) 
    return None

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

def classify_reason(reason_text):
    if not reason_text: return "Misc"
    txt = reason_text.upper()
    
    mm_keys = ['MM', 'MECH', 'BELT', 'ROLL', 'IDLER', 'LINER', 'CHUTE', 'GEAR', 'BEARING', 'PULLEY']
    emd_keys = ['EMD', 'ELEC', 'MOTOR', 'POWER', 'SUPPLY', 'CABLE', 'TRIP', 'FUSE']
    cni_keys = ['C&I', 'CNI', 'SENSOR', 'PROBE', 'SIGNAL', 'PLC', 'COMM', 'ZERO']
    rs_keys = ['C&W', 'WAGON', 'DOOR', 'COUPL', 'RAKE']
    mgr_keys = ['MGR', 'TRACK', 'LOCO', 'DERAIL', 'SLEEPER']
    chem_keys = ['CHEM', 'LAB', 'QUALITY', 'SAMPLE', 'ASH', 'MOISTURE']
    opr_keys = ['OPR', 'OPER', 'CREW', 'SHIFT', 'MANPOWER', 'BUNKER', 'FULL', 'WAIT']

    if any(k in txt for k in mm_keys): return "MM"
    if any(k in txt for k in emd_keys): return "EMD"
    if any(k in txt for k in cni_keys): return "C&I"
    if any(k in txt for k in rs_keys): return "Rolling Stock"
    if any(k in txt for k in mgr_keys): return "MGR"
    if any(k in txt for k in chem_keys): return "Chemistry"
    if any(k in txt for k in opr_keys): return "OPR"
    
    return "Misc"

def parse_demurrage_special(cell_val):
    s = str(cell_val).strip().upper()
    if s in ["", "NAN", "NIL", "-", "NONE"]: return "00:00"
    if ":" in s:
        if re.search(r'\d+:\d+', s): return s
    match = re.search(r'(\d+(\.\d+)?)', s)
    if match:
        try:
            val = float(match.group(1))
            hours = int(val)
            minutes = int((val - hours) * 60)
            return f"{hours:02d}:{minutes:02d}"
        except: pass
    return "00:00"

# ==========================================
# 3. GOOGLE SHEET PARSER (ALL DATA)
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
        
        last_seq_tuple = (0, 0)

        for i in range(len(df_gs)):
            row = df_gs.iloc[i]
            val_b = str(row.iloc[1]).strip()
            val_c = str(row.iloc[2]).strip()
            if not val_b or val_b.lower() == 'nan': continue 
            
            source_val = "Unknown" if (not val_c or val_c.lower() == 'nan') else val_c
            arrival_dt = safe_parse_date(row.iloc[4]) 
            if pd.isnull(arrival_dt): continue
            
            dept_val = str(row.iloc[11]).strip()
            if dept_val.lower() in ['nan', '', 'none']: dept_val = ""
            
            reason_detail = ""
            if i + 1 < len(df_gs):
                next_row = df_gs.iloc[i + 1]
                next_rake_name = str(next_row.iloc[1]).strip()
                if not next_rake_name or next_rake_name.lower() == 'nan':
                    reason_val = str(next_row.iloc[11]).strip()
                    if reason_val.lower() not in ['nan', '', 'none']:
                        reason_detail = reason_val
            
            if dept_val and reason_detail:
                full_remarks_blob = f"{dept_val}|{reason_detail}"
            else:
                full_remarks_blob = ""

            rake_name = val_b
            col_d_val = row.iloc[3]
            wagons, load_type = parse_col_d_wagon_type(col_d_val)

            start_dt = safe_parse_date(row.iloc[5])
            end_dt = safe_parse_date(row.iloc[6])
            if pd.isnull(start_dt): start_dt = arrival_dt 
            
            tippler_timings = {}
            active_tipplers_row = []
            explicit_wagon_counts = {}
            tippler_objs = {} # Store Date Objects

            for t_name, idx in [('T1', 14), ('T2', 15), ('T3', 16), ('T4', 17)]:
                cell_val = row.iloc[idx]
                if pd.notnull(cell_val) and str(cell_val).strip() not in ["", "nan"]:
                    wc = parse_wagon_count_from_cell(cell_val)
                    ts, te = parse_tippler_cell(cell_val, arrival_dt)
                    if pd.isnull(ts) and wc is not None:
                        if pd.notnull(start_dt):
                            ts = start_dt
                            te = end_dt if pd.notnull(end_dt) else start_dt + timedelta(hours=2)
                    elif pd.isnull(ts) and pd.notnull(start_dt):
                        pass

                    if pd.notnull(ts):
                        active_tipplers_row.append(t_name)
                        tippler_timings[f"{t_name} Start"] = format_dt(ts)
                        tippler_timings[f"{t_name} End"] = format_dt(te)
                        tippler_timings[f"{t_name}_Obj_End"] = te
                        
                        # SAVE REAL DATETIME OBJECTS
                        tippler_objs[f"{t_name}_Start_Obj"] = ts
                        tippler_objs[f"{t_name}_End_Obj"] = te
                        
                        if wc is not None: explicit_wagon_counts[t_name] = wc

            is_unplanned = (not active_tipplers_row and load_type != 'BOBR')
            
            seq, rid = parse_last_sequence(rake_name)
            if seq > last_seq_tuple[0] or (seq == last_seq_tuple[0] and rid > last_seq_tuple[1]):
                last_seq_tuple = (seq, rid)

            if is_unplanned:
                unplanned_actuals.append({
                    'Rake': rake_name,  
                    'Coal Source': source_val,
                    'Load Type': load_type,
                    'Wagons': wagons,
                    'Status': 'Pending (G-Sheet)',
                    '_Arrival_DT': arrival_dt,
                    '_Form_Mins': 0,
                    'Optimization Type': 'Auto-Planned (G-Sheet)',
                    'Extra Shunt (Mins)': 0,
                    'is_gs_unplanned': True,
                    '_remarks': full_remarks_blob
                })
                continue 

            t_str_list = []
            wagon_counts_map = {}
            if explicit_wagon_counts:
                for t in active_tipplers_row:
                    cnt = explicit_wagon_counts.get(t, 0)
                    t_str_list.append(f"{t} ({cnt})")
                    wagon_counts_map[f"{t}_Wagons"] = cnt
            elif len(active_tipplers_row) > 0:
                share = wagons // len(active_tipplers_row)
                rem = wagons % len(active_tipplers_row)
                for i, t in enumerate(active_tipplers_row):
                    cnt = share + (1 if i < rem else 0)
                    t_str_list.append(f"{t} ({cnt})")
                    wagon_counts_map[f"{t}_Wagons"] = cnt
            
            used_tipplers_str = ", ".join(t_str_list)

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

            raw_dem_val = row.iloc[10]
            dem_str = parse_demurrage_special(raw_dem_val)

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
                'Tipplers Used': used_tipplers_str,
                'Wait (Tippler)': format_duration_hhmm(start_dt - arrival_dt) if pd.notnull(start_dt) else "",
                'Total Duration': format_duration_hhmm(total_dur),
                'Demurrage': dem_str,
                '_raw_end_dt': end_dt,
                '_raw_tipplers_data': tippler_timings,
                '_raw_wagon_counts': wagon_counts_map,
                '_raw_tipplers': active_tipplers_row,
                '_remarks': full_remarks_blob
            }
            # INJECT HIDDEN OBJECTS
            for t, obj in tippler_objs.items():
                entry[t] = obj # e.g. T1_Start_Obj

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
    if wagons == 0: return ready_time, "", ready_time, {}, timedelta(0), {}
    
    candidates = []
    for t in target_tipplers:
        free_at = tippler_state[t]
        if free_at.tzinfo is None: free_at = IST.localize(free_at)
        prop_start = max(ready_time, free_at)
        effective_start = check_downtime_impact(t, prop_start, downtime_list)
        pred_finish = effective_start + timedelta(hours=wagons / rates[t])
        candidates.append({'id': t, 'rate': rates[t], 'eff_start': effective_start, 'free_at': free_at, 'pred_fin': pred_finish})
    
    candidates.sort(key=lambda x: x['pred_fin'])
    best_solo = candidates[0]
    
    finish_A = best_solo['pred_fin']
    used_list_str = [f"{best_solo['id']} ({wagons})"]
    wagon_counts = {f"{best_solo['id']}_Wagons": wagons}
    
    final_timings = {
        f"{best_solo['id']}_Start": best_solo['eff_start'],
        f"{best_solo['id']}_End": best_solo['pred_fin'],
        f"{best_solo['id']}_Idle": max(timedelta(0), best_solo['eff_start'] - best_solo['free_at'])
    }
    actual_start = best_solo['eff_start']

    if len(candidates) > 1 and wagons > wagons_first_batch:
        prim = candidates[0]
        sec = candidates[1]
        w_first = wagons_first_batch
        w_second = wagons - w_first
        fin_prim_split = prim['eff_start'] + timedelta(hours=w_first / prim['rate'])
        sec_ready_theory = ready_time + timedelta(minutes=inter_tippler_delay)
        prop_start_sec = max(sec_ready_theory, sec['free_at'])
        real_start_sec = check_downtime_impact(sec['id'], prop_start_sec, downtime_list)
        fin_sec_split = real_start_sec + timedelta(hours=w_second / sec['rate'])
        finish_pair = max(fin_prim_split, fin_sec_split)
        
        if finish_pair < finish_A:
            finish_A = finish_pair
            ts = [(prim['id'], w_first), (sec['id'], w_second)]
            ts.sort(key=lambda x: x[0])
            used_list_str = [f"{x[0]} ({x[1]})" for x in ts]
            wagon_counts = {f"{prim['id']}_Wagons": w_first, f"{sec['id']}_Wagons": w_second}
            actual_start = prim['eff_start']
            final_timings = {}
            final_timings[f"{prim['id']}_Start"] = prim['eff_start']
            final_timings[f"{prim['id']}_End"] = fin_prim_split
            final_timings[f"{prim['id']}_Idle"] = max(timedelta(0), prim['eff_start'] - prim['free_at'])
            final_timings[f"{sec['id']}_Start"] = real_start_sec
            final_timings[f"{sec['id']}_End"] = fin_sec_split
            final_timings[f"{sec['id']}_Idle"] = max(timedelta(0), real_start_sec - sec['free_at'])

    return finish_A, ", ".join(used_list_str), actual_start, final_timings, timedelta(0), wagon_counts

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
    if plan_df.empty and df_locked.empty: return pd.DataFrame(), pd.DataFrame(), datetime.now(IST)
    
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
                if '_raw_tipplers' in row:
                    used_list = row['_raw_tipplers']
                    end_val = row['_raw_end_dt']
                    if pd.notnull(end_val) and isinstance(used_list, list):
                        for t in ['T1', 'T2', 'T3', 'T4']:
                            if t in used_list and end_val > tippler_state[t]:
                                tippler_state[t] = end_val

    line_groups = {
        'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []},
        'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}
    }
    
    curr_seq = last_seq_tuple[0]
    curr_id = last_seq_tuple[1]
    assignments = []

    for _, rake in plan_df.iterrows():
        orig_name = str(rake.get('Rake', ''))
        is_gs = rake.get('is_gs_unplanned', False)
        
        if is_gs and '/' in orig_name:
            display_name = orig_name
            s, i = parse_last_sequence(display_name)
            if s > 0: curr_seq, curr_id = s, i
        else:
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
                'Demurrage': "00:00",
                '_raw_wagon_counts': {},
                '_remarks': ""
            }
            for t in ['T1', 'T2', 'T3', 'T4']:
                 row_data[f"{t} Start"] = ""; row_data[f"{t} End"] = ""; row_data[f"{t} Idle"] = ""
            assignments.append(row_data)
            continue

        entry_A = get_line_entry_time('Group_Lines_8_10', rake['_Arrival_DT'], line_groups)
        ready_A = entry_A + timedelta(minutes=s_a)
        fin_A, used_A, start_A, tim_A, _, wag_A = calculate_generic_finish(
            rake['Wagons'], ['T1', 'T2'], ready_A, tippler_state, downtimes, rates, w_batch, w_delay)
        
        entry_B = get_line_entry_time('Group_Line_11', rake['_Arrival_DT'], line_groups)
        ready_B = entry_B + timedelta(minutes=s_b)
        fin_B, used_B, start_B, tim_B, _, wag_B = calculate_generic_finish(
            rake['Wagons'], ['T3', 'T4'], ready_B, tippler_state, downtimes, rates, w_batch, w_delay)
        
        if fin_B < fin_A:
            best_fin, best_used, best_start, best_timings, best_entry, best_ready = fin_B, used_B, start_B, tim_B, entry_B, ready_B
            best_grp, best_line, best_wag = 'Group_Line_11', '11', wag_B
            best_type = "Standard (Fast)"
        else:
            best_fin, best_used, best_start, best_timings, best_entry, best_ready = fin_A, used_A, start_A, tim_A, entry_A, ready_A
            best_grp, best_line, best_wag = 'Group_Lines_8_10', '8/9/10', wag_A
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
            'Demurrage': dem_str,
            '_raw_wagon_counts': best_wag,
            '_remarks': rake.get('_remarks', "")
        }
        for t in ['T1', 'T2', 'T3', 'T4']:
             row_data[f"{t} Start"] = format_dt(best_timings.get(f"{t}_Start", pd.NaT))
             row_data[f"{t} End"] = format_dt(best_timings.get(f"{t}_End", pd.NaT))
             row_data[f"{t} Idle"] = format_duration_hhmm(best_timings.get(f"{t}_Idle", pd.NaT))
        assignments.append(row_data)

    df_sim = pd.DataFrame(assignments)
    
    if not df_locked.empty:
        today_date = datetime.now(IST).date()
        yesterday_date = today_date - timedelta(days=1)
        
        # VISUAL FILTER (TAB 1 ONLY)
        def keep_row(r):
            ad = r['_Arrival_DT'].date()
            if ad >= yesterday_date: return True
            return False 
        df_locked_visible = df_locked[df_locked.apply(keep_row, axis=1)]
        
        cols_to_drop = ['_raw_tipplers_data', '_raw_end_dt', '_raw_tipplers'] + [f"{t}_{x}_Obj" for t in ['T1','T2','T3','T4'] for x in ['Start','End']]
        actuals_clean = df_locked_visible.drop(columns=[c for c in cols_to_drop if c in df_locked_visible.columns], errors='ignore')
        
        final_df_display = pd.concat([actuals_clean, df_sim], ignore_index=True) if not df_sim.empty else actuals_clean
        # FIXED: DO NOT DROP '_Obj' COLUMNS HERE FOR TAB 2
        final_df_all = pd.concat([df_locked, df_sim], ignore_index=True)
    else:
        final_df_display = df_sim
        final_df_all = df_sim

    return final_df_display, final_df_all, sim_start_time

def recalculate_cascade_reactive(df_all, start_filter_dt=None, end_filter_dt=None):
    daily_stats = {} 
    
    for _, row in df_all.iterrows():
        dem_val = str(row['Demurrage']).strip()
        dem_hrs = 0
        if ":" in dem_val: dem_hrs = int(dem_val.split(":")[0])
        elif dem_val.isdigit(): dem_hrs = int(dem_val)
        
        arr_dt = pd.to_datetime(row['_Arrival_DT'])
        if arr_dt.tzinfo is None: arr_dt = IST.localize(arr_dt)
        d_str = arr_dt.strftime('%Y-%m-%d')
        
        if start_filter_dt and arr_dt.date() < start_filter_dt: continue
        if end_filter_dt and arr_dt.date() > end_filter_dt: continue

        if d_str not in daily_stats: 
            daily_stats[d_str] = {'Demurrage': 0, 'All_Reasons': set()}
            for t in ['T1', 'T2', 'T3', 'T4']: 
                daily_stats[d_str][f'{t}_hrs'] = 0.0
                daily_stats[d_str][f'{t}_wag'] = 0.0
        
        daily_stats[d_str]['Demurrage'] += dem_hrs
        
        if dem_hrs > 0:
            rem = str(row.get('_remarks', '')).strip()
            rake_name = str(row.get('Rake', 'Unknown'))
            if '|' in rem:
                dept_part, reason_part = rem.split('|', 1)
            else:
                dept_part, reason_part = rem, ""
            
            if dept_part:
                dept_code = classify_reason(dept_part)
                final_text = reason_part if reason_part else dept_part
                formatted_reason = f"[{rake_name}] - {final_text} ({dept_code})"
                daily_stats[d_str]['All_Reasons'].add(formatted_reason)
        
        wag_map = row.get('_raw_wagon_counts', {})
        if not isinstance(wag_map, dict): wag_map = {}
        
        for t in ['T1', 'T2', 'T3', 'T4']:
            # SMART DATE HANDLING (FIX FOR PAST DATES)
            s_dt = row.get(f"{t}_Start_Obj", pd.NaT)
            e_dt = row.get(f"{t}_End_Obj", pd.NaT)
            
            if pd.isnull(s_dt) or pd.isnull(e_dt):
                # Fallback: Parse string using ARRIVAL DATE context
                start_str = str(row.get(f"{t} Start", ""))
                end_str = str(row.get(f"{t} End", ""))
                if start_str and end_str:
                    s_dt = parse_dt_from_str_smart(start_str, arr_dt)
                    e_dt = parse_dt_from_str_smart(end_str, arr_dt)
            
            if pd.notnull(s_dt) and pd.notnull(e_dt):
                if e_dt < s_dt: e_dt += timedelta(days=1)
                
                total_dur_sec = (e_dt - s_dt).total_seconds()
                total_wagons = wag_map.get(f"{t}_Wagons", 0)
                
                curr = s_dt
                while curr < e_dt:
                    curr_day_str = curr.strftime('%Y-%m-%d')
                    curr_date = curr.date()
                    in_range = True
                    if start_filter_dt and curr_date < start_filter_dt: in_range = False
                    if end_filter_dt and curr_date > end_filter_dt: in_range = False
                    
                    next_midnight = (curr + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    segment_end = min(e_dt, next_midnight)
                    segment_dur_sec = (segment_end - curr).total_seconds()
                    hours = segment_dur_sec / 3600.0
                    
                    if in_range:
                        if curr_day_str not in daily_stats: 
                            daily_stats[curr_day_str] = {'Demurrage': 0, 'All_Reasons': set()}
                            for tx in ['T1', 'T2', 'T3', 'T4']: 
                                daily_stats[curr_day_str][f'{tx}_hrs'] = 0.0
                                daily_stats[curr_day_str][f'{tx}_wag'] = 0.0
                        
                        daily_stats[curr_day_str][f'{t}_hrs'] += hours
                        if total_dur_sec > 0:
                            fraction = segment_dur_sec / total_dur_sec
                            daily_stats[curr_day_str][f'{t}_wag'] += (total_wagons * fraction)
                    
                    curr = segment_end

    output_rows = []
    for d, v in sorted(daily_stats.items()):
        reasons_set = v['All_Reasons']
        major_reasons_str = "\n".join(sorted(reasons_set)) if reasons_set else "-"

        row = {
            'Date': d, 
            'Demurrage': f"{int(v['Demurrage'])} Hours",
            'Major Reasons': major_reasons_str
        }
        for t in ['T1', 'T2', 'T3', 'T4']:
            rate = 0.0
            if v[f'{t}_hrs'] > 0.1: rate = v[f'{t}_wag'] / v[f'{t}_hrs']
            row[f"{t} Rate (W/Hr)"] = f"{rate:.2f}"
        output_rows.append(row)
        
    return pd.DataFrame(output_rows)

def highlight_bobr(row):
    if 'BOBR' in str(row['Load Type']).upper():
        return ['background-color: #FFD700; color: black'] * len(row)
    return [''] * len(row)

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
    
    sim_result, sim_full_result, sim_start_dt = run_full_simulation_initial(
        df_raw, sim_params, df_act, df_unplanned, start_seq
    )
    st.session_state.sim_result = sim_result
    st.session_state.sim_full_result = sim_full_result
    st.session_state.sim_start_dt = sim_start_dt

    if 'sim_result' in st.session_state and not st.session_state.sim_result.empty:
        df_final = st.session_state.sim_result
        df_final['Date_Str'] = df_final['_Arrival_DT'].dt.strftime('%Y-%m-%d')
        unique_dates = sorted(df_final['Date_Str'].unique())

        tab_live, tab_hist = st.tabs(["🚀 Live Schedule", "📜 Historical Analysis"])

        with tab_live:
            col_cfg = {
                "Rake": st.column_config.TextColumn("Rake Name", disabled=True),
                "Coal Source": st.column_config.TextColumn("Source/Mine", disabled=True),
                "Status": st.column_config.TextColumn("Status", help="ACTUAL = From G-Sheet"),
                "Tippler Start Time": st.column_config.TextColumn("Start (dd-HH:MM)"),
                "Finish Unload": st.column_config.TextColumn("Finish (dd-HH:MM)"),
                "Extra Shunt (Mins)": st.column_config.NumberColumn("Ext. Shunt", step=5),
                "_Arrival_DT": None, "_Shunt_Ready_DT": None, "_Form_Mins": None, "Date_Str": None, "_raw_wagon_counts": None, "_remarks": None
            }

            for d in unique_dates:
                st.markdown(f"### 📅 Schedule for {d}")
                day_df = df_final[df_final['Date_Str'] == d].copy()
                day_df.index = np.arange(1, len(day_df) + 1)
                st.dataframe(day_df.style.apply(highlight_bobr, axis=1), use_container_width=True, column_config=col_cfg)

            yest_date = datetime.now(IST).date() - timedelta(days=1)
            daily_stats_df = recalculate_cascade_reactive(st.session_state.sim_full_result, start_filter_dt=yest_date)
            st.markdown("### 📊 Daily Performance & Demurrage Forecast")
            st.dataframe(daily_stats_df, hide_index=True)
            
            st.download_button("📥 Download Final Report", df_final.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins", "Date_Str", "_raw_wagon_counts", "_remarks"]).to_csv(index=False).encode('utf-8'), "optimized_schedule.csv", "text/csv")

        with tab_hist:
            st.subheader("🔍 Past Performance Analysis")
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                view_mode = st.radio("Select View Mode", ["Day View", "Month View", "Custom Range"], horizontal=True)
            
            start_f, end_f = None, None
            with col_h2:
                if view_mode == "Day View":
                    sel_date = st.date_input("Select Date", value=datetime.now(IST).date() - timedelta(days=1))
                    start_f, end_f = sel_date, sel_date
                elif view_mode == "Month View":
                    c_m1, c_m2 = st.columns(2)
                    with c_m1: sel_month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1)
                    with c_m2: sel_year = st.number_input("Year", value=datetime.now().year)
                    import calendar
                    last_day = calendar.monthrange(sel_year, sel_month)[1]
                    start_f = datetime(sel_year, sel_month, 1).date()
                    end_f = datetime(sel_year, sel_month, last_day).date()
                else: 
                    dr = st.date_input("Select Date Range", value=(datetime.now(IST).date()-timedelta(days=7), datetime.now(IST).date()))
                    if isinstance(dr, tuple) and len(dr) == 2: start_f, end_f = dr
            
            if start_f and end_f:
                hist_stats = recalculate_cascade_reactive(st.session_state.sim_full_result, start_filter_dt=start_f, end_filter_dt=end_f)
                st.markdown(f"**Performance Summary ({start_f} to {end_f})**")
                st.dataframe(hist_stats, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                with st.expander("Show Detailed Rake List (Demurrage Only)"):
                    mask = (st.session_state.sim_full_result['_Arrival_DT'].dt.date >= start_f) & (st.session_state.sim_full_result['_Arrival_DT'].dt.date <= end_f)
                    hist_raw = st.session_state.sim_full_result[mask].copy()
                    
                    def has_demurrage(val):
                        s = str(val).strip()
                        return s != "00:00" and s != "0"
                    
                    hist_raw = hist_raw[hist_raw['Demurrage'].apply(has_demurrage)]
                    
                    hist_raw_clean = hist_raw.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins", "Date_Str", "_raw_wagon_counts", "_remarks"] + [f"{t}_{x}_Obj" for t in ['T1','T2','T3','T4'] for x in ['Start','End']], errors='ignore')
                    st.dataframe(hist_raw_clean, use_container_width=True)
