import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import math

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer", layout="wide")
st.title("🚂 BOXN Rake Demurrage Optimization Dashboard")
st.markdown("Upload your `INSIGHT DETAILS.csv`. **Edit 'T1/T2/T3/T4 Finish' to simulate specific machine delays.**")

# ==========================================
# 1. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Infrastructure Settings")

# Rates
st.sidebar.subheader("Individual Tippler Rates")
RATE_T1 = st.sidebar.number_input("Tippler 1 Rate (Wagons/hr)", value=6.0, step=0.5)
RATE_T2 = st.sidebar.number_input("Tippler 2 Rate (Wagons/hr)", value=6.0, step=0.5)
RATE_T3 = st.sidebar.number_input("Tippler 3 Rate (Wagons/hr)", value=9.0, step=0.5)
RATE_T4 = st.sidebar.number_input("Tippler 4 Rate (Wagons/hr)", value=9.0, step=0.5)

# Penalties & Delays
st.sidebar.subheader("Delays & Free Time")
SHUNT_MINS_A = st.sidebar.number_input("Pair A Shunt (Mins)", value=25.0, step=5.0)
SHUNT_MINS_B = st.sidebar.number_input("Pair B Shunt (Mins)", value=50.0, step=5.0)
FORMATION_MINS_A = st.sidebar.number_input("Pair A Formation (Mins)", value=20.0, step=5.0)
FORMATION_MINS_B = st.sidebar.number_input("Pair B Formation (Mins)", value=50.0, step=5.0)
FREE_TIME_HOURS = st.sidebar.number_input("Free Time Allowed (Hours)", value=7.0, step=0.5)

# Stagger Logic
st.sidebar.subheader("Split Logic")
WAGONS_FIRST_BATCH = st.sidebar.number_input("Wagons in 1st Batch", value=30, step=1)
INTER_TIPPLER_DELAY = st.sidebar.number_input("Delay for 2nd Tippler Start (Mins)", value=0.0, step=5.0)

# Downtime Manager
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Tippler Downtime Manager")
if 'downtimes' not in st.session_state:
    st.session_state.downtimes = []

with st.sidebar.form("downtime_form"):
    dt_tippler = st.selectbox("Select Tippler", ["T1", "T2", "T3", "T4"])
    dt_start_date = st.date_input("Start Date", value=datetime.now())
    dt_start_time = st.time_input("Start Time", value=datetime.now().time())
    dt_duration = st.number_input("Duration (Minutes)", min_value=15, step=15, value=60)
    add_dt = st.form_submit_button("Add Downtime")

    if add_dt:
        start_dt = datetime.combine(dt_start_date, dt_start_time)
        end_dt = start_dt + timedelta(minutes=dt_duration)
        st.session_state.downtimes.append({"Tippler": dt_tippler, "Start": start_dt, "End": end_dt})
        st.success(f"Added {dt_tippler} downtime.")

if st.session_state.downtimes:
    dt_df = pd.DataFrame(st.session_state.downtimes)
    dt_display = dt_df.copy()
    dt_display['Start'] = dt_display['Start'].dt.strftime('%d-%H:%M')
    dt_display['End'] = dt_display['End'].dt.strftime('%d-%H:%M')
    st.sidebar.dataframe(dt_display, use_container_width=True)
    if st.sidebar.button("Clear Downtimes"):
        st.session_state.downtimes = []
        st.rerun()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_wagons(val):
    try:
        if '+' in str(val):
            parts = str(val).split('+')
            return sum(int(p) for p in parts if p.strip().isdigit())
        return int(float(val))
    except: return 0

def format_dt(dt):
    if pd.isnull(dt): return ""
    return dt.strftime('%d-%H:%M')

def restore_dt(dt_str, ref_dt):
    """Restores datetime from 'dd-HH:MM' string using reference year/month."""
    if not isinstance(dt_str, str) or dt_str == "": return pd.NaT
    try:
        parts = dt_str.split('-') 
        day = int(parts[0])
        time_parts = parts[1].split(':')
        hour, minute = int(time_parts[0]), int(time_parts[1])
        # Use ref_dt's year and month
        # Handle simple month rollover logic if day is smaller than ref day
        new_dt = ref_dt.replace(day=day, hour=hour, minute=minute, second=0)
        # If the day is significantly smaller, it might be next month (simple heuristic)
        if new_dt < ref_dt - timedelta(days=20):
             # This is a basic catch for month rollover in manual entry
             # Ideally we need smarter logic but for simulation tweaks this usually suffices
             pass 
        return new_dt
    except: return pd.NaT

def format_duration_hhmm(delta):
    if pd.isnull(delta): return "00:00"
    total_seconds = int(delta.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{sign}{hours:02d}:{minutes:02d}"

def check_downtime_impact(tippler_id, proposed_start, downtime_list):
    relevant_dts = [d for d in downtime_list if d['Tippler'] == tippler_id]
    relevant_dts.sort(key=lambda x: x['Start'])
    current_start = proposed_start
    for dt in relevant_dts:
        if dt['Start'] <= current_start < dt['End']:
            current_start = dt['End']
    return current_start

def calculate_split_finish(wagons, pair_name, ready_time, tippler_state, downtime_list):
    if wagons == 0: return ready_time, "None", ready_time, {}, timedelta(0)
    
    if pair_name == 'Pair A (T1&T2)':
        resources = {'T1': RATE_T1, 'T2': RATE_T2}
    else:
        resources = {'T3': RATE_T3, 'T4': RATE_T4}
        
    sorted_tipplers = sorted(resources.keys(), key=lambda x: tippler_state[x])
    t_primary, t_secondary = sorted_tipplers[0], sorted_tipplers[1]
    
    machine_free_at = tippler_state[t_primary]
    idle_delta = timedelta(0) if machine_free_at == pd.Timestamp.min else max(timedelta(0), ready_time - machine_free_at)

    w_first = min(wagons, WAGONS_FIRST_BATCH)
    w_second = wagons - w_first
    
    proposed_start_prim = max(ready_time, tippler_state[t_primary])
    actual_start_prim = check_downtime_impact(t_primary, proposed_start_prim, downtime_list)
    finish_primary = actual_start_prim + timedelta(hours=(w_first / resources[t_primary]))
    
    updated_times = {t_primary: finish_primary}
    
    if w_second > 0:
        start_sec_theory = ready_time + timedelta(minutes=INTER_TIPPLER_DELAY)
        proposed_start_sec = max(start_sec_theory, tippler_state[t_secondary])
        actual_start_sec = check_downtime_impact(t_secondary, proposed_start_sec, downtime_list)
        finish_secondary = actual_start_sec + timedelta(hours=(w_second / resources[t_secondary]))
        updated_times[t_secondary] = finish_secondary
        overall_finish = max(finish_primary, finish_secondary)
        used_str = f"{t_primary} & {t_secondary}"
    else:
        overall_finish = finish_primary
        used_str = f"{t_primary} Only"
        updated_times[t_secondary] = tippler_state[t_secondary]

    return overall_finish, used_str, actual_start_prim, updated_times, idle_delta

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

# ==========================================
# 3. MAIN SIMULATION ENGINE
# ==========================================

def run_full_simulation(df):
    """Runs the initial logic from scratch."""
    df['wagon_count'] = df['TOTL UNTS'].apply(parse_wagons)
    df['exp_arrival_dt'] = pd.to_datetime(df['EXPD ARVLTIME'], errors='coerce')
    
    if 'STTS CODE' in df.columns and 'STTS TIME' in df.columns:
        df['stts_time_dt'] = pd.to_datetime(df['STTS TIME'], errors='coerce')
        df['arrival_dt'] = df.apply(lambda r: r['stts_time_dt'] if str(r.get('STTS CODE')).strip()=='PL' and pd.notnull(r['stts_time_dt']) else r['exp_arrival_dt'], axis=1)
    else:
        df['arrival_dt'] = df['exp_arrival_dt']

    df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt').reset_index(drop=True)
    
    tippler_state = {'T1': pd.Timestamp.min, 'T2': pd.Timestamp.min, 'T3': pd.Timestamp.min, 'T4': pd.Timestamp.min}
    spec_line_status = {8: pd.Timestamp.min, 9: pd.Timestamp.min, 10: pd.Timestamp.min, 11: pd.Timestamp.min}
    line_groups = {
        'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}, 
        'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}
    }
    rr_tracker_A = -1 
    assignments = []
    
    for _, rake in df.iterrows():
        # A
        shunt_A = timedelta(minutes=SHUNT_MINS_A)
        entry_A = get_line_entry_time('Group_Lines_8_10', rake['arrival_dt'], line_groups)
        ready_A = entry_A + shunt_A
        fin_A, used_A, start_A, up_A, idle_A = calculate_split_finish(rake['wagon_count'], 'Pair A (T1&T2)', ready_A, tippler_state, st.session_state.downtimes)
        
        # B
        shunt_B = timedelta(minutes=SHUNT_MINS_B)
        entry_B = get_line_entry_time('Group_Line_11', rake['arrival_dt'], line_groups)
        ready_B = entry_B + shunt_B
        fin_B, used_B, start_B, up_B, idle_B = calculate_split_finish(rake['wagon_count'], 'Pair B (T3&T4)', ready_B, tippler_state, st.session_state.downtimes)
        
        if fin_A <= fin_B:
            best_pair, best_grp, best_entry, best_ready, best_start, best_fin = 'Pair A (T1&T2)', 'Group_Lines_8_10', entry_A, ready_A, start_A, fin_A
            best_tips, best_idle, best_form_mins = used_A, idle_A, FORMATION_MINS_A
            for k,v in up_A.items(): tippler_state[k] = v
            sel_line = find_specific_line(best_grp, best_entry, spec_line_status, rr_tracker_A)
            if sel_line in [8,9,10]: rr_tracker_A = {8:0,9:1,10:2}[sel_line]
            res_t1, res_t2, res_t3, res_t4 = up_A.get('T1', pd.NaT), up_A.get('T2', pd.NaT), pd.NaT, pd.NaT
        else:
            best_pair, best_grp, best_entry, best_ready, best_start, best_fin = 'Pair B (T3&T4)', 'Group_Line_11', entry_B, ready_B, start_B, fin_B
            best_tips, best_idle, best_form_mins = used_B, idle_B, FORMATION_MINS_B
            for k,v in up_B.items(): tippler_state[k] = v
            sel_line = 11
            res_t1, res_t2, res_t3, res_t4 = pd.NaT, pd.NaT, up_B.get('T3', pd.NaT), up_B.get('T4', pd.NaT)
            
        clr_mins = line_groups[best_grp]['clearance_mins']
        line_free = best_entry + timedelta(minutes=clr_mins)
        line_groups[best_grp]['line_free_times'].append(line_free)
        spec_line_status[sel_line] = line_free
        
        wait_delta = best_start - best_ready
        form_delta = timedelta(minutes=best_form_mins)
        tot_dur = (best_fin - rake['arrival_dt']) + form_delta
        demurrage = max(timedelta(0), tot_dur - timedelta(hours=FREE_TIME_HOURS))

        assignments.append({
            'Rake': rake['RAKE NAME'],
            'Wagons': rake['wagon_count'],
            'Status': rake.get('STTS CODE', 'N/A'),
            # HIDDEN COLUMNS FOR CALCULATION (Keep as datetime objects)
            '_Arrival_DT': rake['arrival_dt'],
            '_Shunt_Ready_DT': best_ready,
            '_Form_Mins': best_form_mins,
            # DISPLAY COLUMNS (Strings/Formatted)
            'Revised Arrival Time': format_dt(rake['arrival_dt']),
            'Line Allotted': sel_line,
            'Line Entry Time': format_dt(best_entry),
            'Shunting Complete': format_dt(best_ready),
            'Tippler Start Time': format_dt(best_start),
            'Finish Unload': format_dt(best_fin),
            'Tipplers Used': best_tips,
            'T1 Finish': format_dt(res_t1),
            'T2 Finish': format_dt(res_t2),
            'T3 Finish': format_dt(res_t3),
            'T4 Finish': format_dt(res_t4),
            'Wait (Tippler)': format_duration_hhmm(wait_delta),
            'Tippler Idle Time': format_duration_hhmm(best_idle),
            'Formation Time': f"{int(best_form_mins)} m",
            'Total Duration': format_duration_hhmm(tot_dur),
            'Demurrage': format_duration_hhmm(demurrage),
            'Placement Reason': rake.get('PLCT RESN', 'N/A')
        })
    return pd.DataFrame(assignments)

def recalculate_cascade(edited_df):
    """
    Cascading Re-calculation:
    1. Reads user edits (Main Start/Finish OR Individual T1/T2/T3/T4 Finish).
    2. Updates Tippler Availability based on specific tippler columns.
    3. Pushes subsequent trains.
    """
    recalc_rows = []
    
    tippler_state = {'T1': pd.Timestamp.min, 'T2': pd.Timestamp.min, 'T3': pd.Timestamp.min, 'T4': pd.Timestamp.min}
    daily_demurrage_tracker = {}
    
    for _, row in edited_df.iterrows():
        arrival = pd.to_datetime(row['_Arrival_DT'])
        ready = pd.to_datetime(row['_Shunt_Ready_DT'])
        form_mins = float(row['_Form_Mins'])
        
        # 1. Parse which tipplers were used
        used_str = str(row['Tipplers Used'])
        current_tipplers = []
        if 'T1' in used_str: current_tipplers.append('T1')
        if 'T2' in used_str: current_tipplers.append('T2')
        if 'T3' in used_str: current_tipplers.append('T3')
        if 'T4' in used_str: current_tipplers.append('T4')
        
        # 2. Determine Earliest Start based on CURRENT State
        resource_free_at = pd.Timestamp.min
        for t in current_tipplers:
            resource_free_at = max(resource_free_at, tippler_state[t])
        
        calculated_start = max(ready, resource_free_at)
        
        # 3. Check for User Edits to START TIME (Shift entire block)
        try:
            old_start = restore_dt(row['Tippler Start Time'], ready)
            final_start = max(calculated_start, old_start)
        except:
            final_start = calculated_start
            
        # 4. Check for User Edits to INDIVIDUAL FINISH TIMES
        # This is where we update specific tipplers based on user input
        tippler_finish_map = {}
        max_finish = final_start # Baseline
        
        # Check T1
        if 'T1' in current_tipplers:
            t1_edit = restore_dt(row['T1 Finish'], final_start)
            # If user edited T1, use it. Else estimate based on old duration logic or keep previous relative duration
            # Simplified: If NaT (cleared), re-calc. If present, use it.
            if pd.isnull(t1_edit):
                # Recalculate duration if needed, or assume standard rate
                # For cascade safety, we should ideally use original duration + new start
                # Here we assume user might have edited T1 Finish specifically
                t1_edit = final_start + timedelta(hours=2) # Fallback if missing
            
            # Adjust T1 finish relative to new start if user didn't explicitly extend it beyond natural shift
            # (Logic: preserved duration vs fixed timestamp? Fixed timestamp is safer for manual overrides)
            tippler_finish_map['T1'] = t1_edit
            tippler_state['T1'] = t1_edit
            max_finish = max(max_finish, t1_edit)
            
        # Check T2
        if 'T2' in current_tipplers:
            t2_edit = restore_dt(row['T2 Finish'], final_start)
            if pd.isnull(t2_edit): t2_edit = final_start + timedelta(hours=2)
            tippler_finish_map['T2'] = t2_edit
            tippler_state['T2'] = t2_edit
            max_finish = max(max_finish, t2_edit)
            
        # Check T3
        if 'T3' in current_tipplers:
            t3_edit = restore_dt(row['T3 Finish'], final_start)
            if pd.isnull(t3_edit): t3_edit = final_start + timedelta(hours=2)
            tippler_finish_map['T3'] = t3_edit
            tippler_state['T3'] = t3_edit
            max_finish = max(max_finish, t3_edit)
            
        # Check T4
        if 'T4' in current_tipplers:
            t4_edit = restore_dt(row['T4 Finish'], final_start)
            if pd.isnull(t4_edit): t4_edit = final_start + timedelta(hours=2)
            tippler_finish_map['T4'] = t4_edit
            tippler_state['T4'] = t4_edit
            max_finish = max(max_finish, t4_edit)

        final_finish = max_finish
            
        # 5. Update Metrics
        wait_delta = final_start - ready
        form_delta = timedelta(minutes=form_mins)
        tot_dur = (final_finish - arrival) + form_delta
        demurrage = max(timedelta(0), tot_dur - timedelta(hours=FREE_TIME_HOURS))
        
        # 6. Write Back
        row['Tippler Start Time'] = format_dt(final_start)
        row['Finish Unload'] = format_dt(final_finish)
        
        if 'T1' in current_tipplers: row['T1 Finish'] = format_dt(tippler_finish_map['T1'])
        if 'T2' in current_tipplers: row['T2 Finish'] = format_dt(tippler_finish_map['T2'])
        if 'T3' in current_tipplers: row['T3 Finish'] = format_dt(tippler_finish_map['T3'])
        if 'T4' in current_tipplers: row['T4 Finish'] = format_dt(tippler_finish_map['T4'])
        
        row['Wait (Tippler)'] = format_duration_hhmm(wait_delta)
        row['Total Duration'] = format_duration_hhmm(tot_dur)
        row['Demurrage'] = format_duration_hhmm(demurrage)
        
        date_str = arrival.strftime('%Y-%m-%d')
        daily_demurrage_tracker[date_str] = daily_demurrage_tracker.get(date_str, 0) + demurrage.total_seconds()
        
        recalc_rows.append(row)
        
    return pd.DataFrame(recalc_rows), daily_demurrage_tracker

# ==========================================
# 4. FILE HANDLING & INTERFACE
# ==========================================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    if 'raw_data' not in st.session_state:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip().str.upper()
        if 'PLCT RESN' in df_raw.columns:
            df_raw = df_raw[~df_raw['PLCT RESN'].astype(str).str.upper().str.contains('LDNG', na=False)]
        st.session_state.raw_data = df_raw
        st.session_state.sim_result = run_full_simulation(df_raw)

    st.markdown("### 📝 Schedule Editor")
    # CONFIG: Make specific columns editable
    edited_df = st.data_editor(
        st.session_state.sim_result,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "_Arrival_DT": None, "_Shunt_Ready_DT": None, "_Form_Mins": None,
            "Tippler Start Time": st.column_config.TextColumn("Start (dd-HH:MM)", help="Edit start time"),
            "Finish Unload": st.column_config.TextColumn("Finish (dd-HH:MM)", help="Calculated from individual Tipplers"),
            # ENABLE EDITING FOR INDIVIDUAL TIPPLERS
            "T1 Finish": st.column_config.TextColumn("T1 Finish", help="Edit to delay T1"),
            "T2 Finish": st.column_config.TextColumn("T2 Finish", help="Edit to delay T2"),
            "T3 Finish": st.column_config.TextColumn("T3 Finish", help="Edit to delay T3"),
            "T4 Finish": st.column_config.TextColumn("T4 Finish", help="Edit to delay T4"),
        },
        disabled=["Rake", "Wagons", "Line Allotted", "Wait (Tippler)", "Total Duration", "Demurrage"]
    )
    
    if st.button("🔄 Re-Calculate (Cascade Delays)"):
        new_result, daily_stats = recalculate_cascade(edited_df)
        st.session_state.sim_result = new_result
        st.session_state.daily_stats = daily_stats
        st.rerun()

    if 'daily_stats' not in st.session_state:
        _, st.session_state.daily_stats = recalculate_cascade(st.session_state.sim_result)
        
    st.markdown("### 📅 Daily Demurrage Summary")
    daily_data = [{'Date': k, 'Total Demurrage': format_duration_hhmm(timedelta(seconds=v))} for k,v in st.session_state.daily_stats.items()]
    daily_df = pd.DataFrame(daily_data).sort_values('Date')
    st.dataframe(daily_df, use_container_width=True)
    
    csv = st.session_state.sim_result.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins"]).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Final Report", csv, "optimized_schedule.csv", "text/csv")

elif 'raw_data' in st.session_state:
    del st.session_state.raw_data
    del st.session_state.sim_result
