import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import math

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer", layout="wide")
st.title("🚂 BOXN Rake Demurrage Optimization Dashboard")
st.markdown("Upload your `INSIGHT DETAILS.csv`. **Edit any time value -> The schedule automatically updates.**")

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
    """
    Robustly parses 'dd-HH:MM' string.
    Falls back to ref_dt (usually 'Ready Time') if parsing fails.
    """
    if not isinstance(dt_str, str) or dt_str.strip() == "": return pd.NaT
    try:
        parts = dt_str.split('-') 
        day = int(parts[0])
        time_parts = parts[1].split(':')
        hour, minute = int(time_parts[0]), int(time_parts[1])
        
        new_dt = ref_dt.replace(day=day, hour=hour, minute=minute, second=0)
        
        # Simple month boundary handling
        if day < ref_dt.day - 15: 
            new_dt = new_dt + pd.DateOffset(months=1)
        elif day > ref_dt.day + 15:
            new_dt = new_dt - pd.DateOffset(months=1)
            
        return new_dt
    except: 
        return pd.NaT

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
    changed = True
    while changed:
        changed = False
        for dt in relevant_dts:
            if dt['Start'] <= current_start < dt['End']:
                current_start = dt['End']
                changed = True
    return current_start

# ==========================================
# 3. CORE SIMULATION ENGINES
# ==========================================

def calculate_split_finish(wagons, pair_name, ready_time, tippler_state, downtime_list, 
                           rate_t1, rate_t2, rate_t3, rate_t4, 
                           wagons_first_batch, inter_tippler_delay):
    # Standard logic for initial run (Physics only)
    if wagons == 0: return ready_time, "None", ready_time, {}, timedelta(0)
    
    if pair_name == 'Pair A (T1&T2)':
        resources = {'T1': rate_t1, 'T2': rate_t2}
    else:
        resources = {'T3': rate_t3, 'T4': rate_t4}
        
    sorted_tipplers = sorted(resources.keys(), key=lambda x: tippler_state[x])
    t_primary, t_secondary = sorted_tipplers[0], sorted_tipplers[1]
    
    machine_free_at = tippler_state[t_primary]
    idle_delta = timedelta(0) if machine_free_at == pd.Timestamp.min else max(timedelta(0), ready_time - machine_free_at)

    w_first = min(wagons, wagons_first_batch)
    w_second = wagons - w_first
    
    proposed_start_prim = max(ready_time, tippler_state[t_primary])
    actual_start_prim = check_downtime_impact(t_primary, proposed_start_prim, downtime_list)
    finish_primary = actual_start_prim + timedelta(hours=(w_first / resources[t_primary]))
    
    updated_times = {t_primary: finish_primary}
    
    if w_second > 0:
        start_sec_theory = ready_time + timedelta(minutes=inter_tippler_delay)
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

def run_full_simulation_initial(df, params):
    """
    Runs the simulation from scratch using only raw constraints (Start from Zero).
    This creates the 'baseline' optimized schedule.
    """
    r_t1, r_t2, r_t3, r_t4 = params['rt1'], params['rt2'], params['rt3'], params['rt4']
    s_a, s_b = params['sa'], params['sb']
    f_a, f_b = params['fa'], params['fb']
    w_batch, w_delay = params['wb'], params['wd']
    ft_hours = params['ft']
    downtimes = params['downtimes']

    df = df.copy()
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
        shunt_A = timedelta(minutes=s_a)
        entry_A = get_line_entry_time('Group_Lines_8_10', rake['arrival_dt'], line_groups)
        ready_A = entry_A + shunt_A
        fin_A, used_A, start_A, up_A, idle_A = calculate_split_finish(
            rake['wagon_count'], 'Pair A (T1&T2)', ready_A, tippler_state, downtimes,
            r_t1, r_t2, r_t3, r_t4, w_batch, w_delay
        )
        
        shunt_B = timedelta(minutes=s_b)
        entry_B = get_line_entry_time('Group_Line_11', rake['arrival_dt'], line_groups)
        ready_B = entry_B + shunt_B
        fin_B, used_B, start_B, up_B, idle_B = calculate_split_finish(
            rake['wagon_count'], 'Pair B (T3&T4)', ready_B, tippler_state, downtimes,
            r_t1, r_t2, r_t3, r_t4, w_batch, w_delay
        )
        
        if fin_A <= fin_B:
            best_pair, best_grp, best_entry, best_ready, best_start, best_fin = 'Pair A (T1&T2)', 'Group_Lines_8_10', entry_A, ready_A, start_A, fin_A
            best_tips, best_idle, best_form_mins = used_A, idle_A, f_a
            for k,v in up_A.items(): tippler_state[k] = v
            sel_line = find_specific_line(best_grp, best_entry, spec_line_status, rr_tracker_A)
            if sel_line in [8,9,10]: rr_tracker_A = {8:0,9:1,10:2}[sel_line]
            res_t1, res_t2, res_t3, res_t4 = up_A.get('T1', pd.NaT), up_A.get('T2', pd.NaT), pd.NaT, pd.NaT
        else:
            best_pair, best_grp, best_entry, best_ready, best_start, best_fin = 'Pair B (T3&T4)', 'Group_Line_11', entry_B, ready_B, start_B, fin_B
            best_tips, best_idle, best_form_mins = used_B, idle_B, f_b
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
        demurrage = max(timedelta(0), tot_dur - timedelta(hours=ft_hours))

        assignments.append({
            'Rake': rake['RAKE NAME'],
            'Wagons': rake['wagon_count'],
            'Status': rake.get('STTS CODE', 'N/A'),
            '_Arrival_DT': rake['arrival_dt'],
            '_Shunt_Ready_DT': best_ready,
            '_Form_Mins': best_form_mins,
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

def recalculate_cascade_reactive(edited_df, free_time_hours):
    """
    REACTIVE SIMULATION:
    1. Reads the EDITED dataframe top-to-bottom.
    2. Maintains a running state of Tippler Availability.
    3. For each row:
       a. Calculates earliest VALID start based on Tippler Availability (Physics).
       b. Checks if the user EDITED the Start Time manually.
          - If User Start > Valid Start (Delay), we ACCEPT it.
          - If User Start < Valid Start (Impossible), we REJECT it and enforce Valid Start.
       c. Checks if the user EDITED Finish Time manually.
          - If yes, we LOCK the tippler until that time.
          - If no, we calculate finish based on duration + new start.
       d. Updates Tippler Availability for the NEXT row.
    4. Updates Demurrage stats.
    """
    recalc_rows = []
    tippler_state = {'T1': pd.Timestamp.min, 'T2': pd.Timestamp.min, 'T3': pd.Timestamp.min, 'T4': pd.Timestamp.min}
    daily_demurrage_tracker = {}
    
    for _, row in edited_df.iterrows():
        # Restore basic data
        arrival = pd.to_datetime(row['_Arrival_DT'])
        ready = pd.to_datetime(row['_Shunt_Ready_DT'])
        form_mins = float(row['_Form_Mins'])
        
        used_str = str(row['Tipplers Used'])
        current_tipplers = []
        if 'T1' in used_str: current_tipplers.append('T1')
        if 'T2' in used_str: current_tipplers.append('T2')
        if 'T3' in used_str: current_tipplers.append('T3')
        if 'T4' in used_str: current_tipplers.append('T4')
        
        # 1. Physics Check: When are resources free?
        resource_free_at = pd.Timestamp.min
        for t in current_tipplers:
            resource_free_at = max(resource_free_at, tippler_state[t])
        
        # Earliest Possible Start
        valid_start = max(ready, resource_free_at)
        
        # 2. Check User Manual Start (Did they type something different?)
        # We assume the value in 'Tippler Start Time' column is what the user *wants*.
        user_start_input = restore_dt(row['Tippler Start Time'], ready)
        
        # Logic: If user input is valid and LATER than physics allow, use it (Delay).
        # Otherwise, snap back to physics.
        if pd.notnull(user_start_input):
            final_start = max(valid_start, user_start_input)
        else:
            final_start = valid_start
            
        # 3. Determine Finish Times (Snap subsequent trains)
        tippler_finish_map = {}
        max_finish = final_start 
        
        for t_id in ['T1', 'T2', 'T3', 'T4']:
            if t_id in current_tipplers:
                col_name = f"{t_id} Finish"
                
                # Check what is currently in the cell (User edit OR previous calc)
                t_val_in_cell = restore_dt(row[col_name], final_start)
                
                if pd.notnull(t_val_in_cell):
                    # We have a value. Is it a manual override or just the old value shifting?
                    # Key decision: We prioritize preserving DURATION if start moved, 
                    # UNLESS the user explicitly typed a finish time that implies a new duration.
                    
                    # Since we can't distinguish "User Typed" vs "Auto Filled" easily here without diffing,
                    # We adopt a "Sticky Finish" strategy: 
                    # If the user sets a Finish time, we try to honor it.
                    # BUT, if 'final_start' moved forward (delayed), we must shift the finish too to keep duration positive.
                    
                    # Heuristic: Calculate implied duration from the cell values.
                    # If duration is reasonable (>0), use start + duration.
                    # This effectively "shifts" the block if start moves.
                    
                    # HOWEVER, if the user EDITED the finish time to be later, we want that new duration.
                    
                    # Safest Approach for "Automatic Simulation":
                    # We assume the value in the cell is the 'Target Finish'.
                    # If Target Finish < Final Start (Impossible), we shift it.
                    if t_val_in_cell < final_start:
                         # Start pushed past finish. Maintain 2hr duration or previous logic
                         t_final = final_start + timedelta(hours=2) 
                    else:
                         t_final = t_val_in_cell
                else:
                    t_final = final_start + timedelta(hours=2)

                tippler_finish_map[t_id] = t_final
                tippler_state[t_id] = t_final # Lock resource
                max_finish = max(max_finish, t_final)

        final_finish = max_finish
            
        # 4. Metrics
        wait_delta = final_start - ready
        form_delta = timedelta(minutes=form_mins)
        tot_dur = (final_finish - arrival) + form_delta
        demurrage = max(timedelta(0), tot_dur - timedelta(hours=free_time_hours))
        
        # 5. Write Updates
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
# 4. SIDEBAR PARAMS
# ==========================================
st.sidebar.header("⚙️ Infrastructure Settings")
sim_params = {}
sim_params['rt1'] = st.sidebar.number_input("Tippler 1 Rate", value=6.0, step=0.5)
sim_params['rt2'] = st.sidebar.number_input("Tippler 2 Rate", value=6.0, step=0.5)
sim_params['rt3'] = st.sidebar.number_input("Tippler 3 Rate", value=9.0, step=0.5)
sim_params['rt4'] = st.sidebar.number_input("Tippler 4 Rate", value=9.0, step=0.5)
sim_params['sa'] = st.sidebar.number_input("Pair A Shunt (Mins)", value=25.0, step=5.0)
sim_params['sb'] = st.sidebar.number_input("Pair B Shunt (Mins)", value=50.0, step=5.0)
sim_params['fa'] = st.sidebar.number_input("Pair A Formation (Mins)", value=20.0, step=5.0)
sim_params['fb'] = st.sidebar.number_input("Pair B Formation (Mins)", value=50.0, step=5.0)
sim_params['ft'] = st.sidebar.number_input("Free Time (Hours)", value=7.0, step=0.5)
sim_params['wb'] = st.sidebar.number_input("Wagons 1st Batch", value=30, step=1)
sim_params['wd'] = st.sidebar.number_input("Delay 2nd Tippler (Mins)", value=0.0, step=5.0)

# DOWNTIME
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Tippler Downtime Manager")
if 'downtimes' not in st.session_state: st.session_state.downtimes = []
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
        # Force Full Reload
        if 'raw_data' in st.session_state:
            sim_params['downtimes'] = st.session_state.downtimes
            st.session_state.sim_result = run_full_simulation_initial(st.session_state.raw_data, sim_params)
            st.rerun()

if st.session_state.downtimes:
    dt_df = pd.DataFrame(st.session_state.downtimes)
    st.sidebar.dataframe(dt_df[['Tippler', 'Start', 'End']], use_container_width=True)
    if st.sidebar.button("Clear Downtimes"):
        st.session_state.downtimes = []
        if 'raw_data' in st.session_state:
            sim_params['downtimes'] = []
            st.session_state.sim_result = run_full_simulation_initial(st.session_state.raw_data, sim_params)
        st.rerun()

sim_params['downtimes'] = st.session_state.downtimes

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip().str.upper()
        if 'PLCT RESN' in df_raw.columns:
            df_raw = df_raw[~df_raw['PLCT RESN'].astype(str).str.upper().str.contains('LDNG', na=False)]
        
        st.session_state.raw_data = df_raw
        st.session_state.last_uploaded_file = uploaded_file
        # Initial Physics Run
        st.session_state.sim_result = run_full_simulation_initial(df_raw, sim_params)
        _, st.session_state.daily_stats = recalculate_cascade_reactive(st.session_state.sim_result, sim_params['ft'])

if 'raw_data' in st.session_state:
    st.markdown("### 📝 Schedule Editor")
    
    # --- AUTO-UPDATE LOOP ---
    # 1. Render Table with current state
    edited_df = st.data_editor(
        st.session_state.sim_result,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "_Arrival_DT": None, "_Shunt_Ready_DT": None, "_Form_Mins": None,
            "Tippler Start Time": st.column_config.TextColumn("Start (dd-HH:MM)", help="Edit to delay"),
            "Finish Unload": st.column_config.TextColumn("Finish (dd-HH:MM)", help="Edit to extend"),
            "T1 Finish": st.column_config.TextColumn("T1 Finish"),
            "T2 Finish": st.column_config.TextColumn("T2 Finish"),
            "T3 Finish": st.column_config.TextColumn("T3 Finish"),
            "T4 Finish": st.column_config.TextColumn("T4 Finish"),
        },
        disabled=["Rake", "Wagons", "Line Allotted", "Wait (Tippler)", "Total Duration", "Demurrage"],
        key="data_editor"
    )

    # 2. Reactive Check
    # If the user edits the table, 'edited_df' reflects changes.
    # We compare it to 'sim_result'. If different (user input), we recalc and rerun.
    # BUT we must avoid infinite loops. The recalc logic ensures consistency.
    
    # We perform the calculation on the edited data
    new_result, daily_stats = recalculate_cascade_reactive(edited_df, sim_params['ft'])
    
    # 3. Detect Change
    # To detect if we need to update the screen, we check if the new calculation
    # differs from what is currently displayed in the editor state.
    # Simple check: Does the 'new_result' match 'st.session_state.sim_result'?
    
    # We ignore the hidden columns for comparison to be safe
    cols_to_compare = ['Tippler Start Time', 'Finish Unload', 'Wait (Tippler)', 'Demurrage', 
                       'T1 Finish', 'T2 Finish', 'T3 Finish', 'T4 Finish']
    
    try:
        is_changed = False
        # Deep compare of specific relevant columns
        for col in cols_to_compare:
            if not new_result[col].equals(st.session_state.sim_result[col]):
                is_changed = True
                break
        
        if is_changed:
            st.session_state.sim_result = new_result
            st.session_state.daily_stats = daily_stats
            st.rerun() # Forces the editor to reload with the new calculated values (snapping)
            
    except Exception as e:
        pass # Handle potential nan comparison issues gracefully

    # 4. Display Stats (Always reflects current state)
    st.markdown("### 📅 Daily Demurrage Summary")
    if 'daily_stats' in st.session_state:
        daily_data = [{'Date': k, 'Total Demurrage': format_duration_hhmm(timedelta(seconds=v))} for k,v in st.session_state.daily_stats.items()]
        if daily_data:
            st.dataframe(pd.DataFrame(daily_data).sort_values('Date'), use_container_width=True)
        else:
            st.info("No Demurrage Incurred.")
            
    csv = st.session_state.sim_result.drop(columns=["_Arrival_DT", "_Shunt_Ready_DT", "_Form_Mins"]).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Final Report", csv, "optimized_schedule.csv", "text/csv")
