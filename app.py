import streamlit as st
import pandas as pd
from datetime import timedelta
import math

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer", layout="wide")
st.title("🚂 BOXN Rake Demurrage Optimization Dashboard")
st.markdown("Upload your `INSIGHT DETAILS.csv` to run the simulation based on your offline logic.")

# ==========================================
# 1. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Infrastructure Settings")

# Demurrage Settings (NEW)
st.sidebar.subheader("Demurrage Rules")
FREE_TIME_HOURS = st.sidebar.number_input("Free Time Allowed (Hours)", value=7.0, step=0.5)

# Unloading Rates
st.sidebar.subheader("Individual Tippler Rates")
RATE_T1 = st.sidebar.number_input("Tippler 1 Rate (Wagons/hr)", value=6.0, step=0.5)
RATE_T2 = st.sidebar.number_input("Tippler 2 Rate (Wagons/hr)", value=6.0, step=0.5)
RATE_T3 = st.sidebar.number_input("Tippler 3 Rate (Wagons/hr)", value=9.0, step=0.5)
RATE_T4 = st.sidebar.number_input("Tippler 4 Rate (Wagons/hr)", value=9.0, step=0.5)

# Shunting Penalties
st.sidebar.subheader("Shunting Delays")
SHUNT_MINS_A = st.sidebar.number_input("Pair A Initial Shunt (Mins)", value=25.0, step=5.0)
SHUNT_MINS_B = st.sidebar.number_input("Pair B Initial Shunt (Mins)", value=50.0, step=5.0)

# Rake Formation Time
st.sidebar.subheader("Rake Formation Time")
FORMATION_MINS_A = st.sidebar.number_input("Pair A Formation (Mins)", value=20.0, step=5.0)
FORMATION_MINS_B = st.sidebar.number_input("Pair B Formation (Mins)", value=50.0, step=5.0)

# Stagger Logic
st.sidebar.subheader("Split Logic")
WAGONS_FIRST_BATCH = st.sidebar.number_input("Wagons in 1st Batch", value=30, step=1)
INTER_TIPPLER_DELAY = st.sidebar.number_input("Delay for 2nd Tippler Start (Mins)", value=0.0, step=5.0)

# Line Clearance Constraints
st.sidebar.info(f"Line 8-10 Clearance: 50 mins (Parallel Processing)\nLine 11 Clearance: 100 mins (Parallel Processing)")

# ==========================================
# 2. CALCULATION LOGIC
# ==========================================

def calculate_split_finish(wagons, pair_name, ready_time, tippler_state):
    """
    Calculates finish time using DYNAMIC ASSIGNMENT.
    """
    if wagons == 0: 
        return ready_time, "None", ready_time, {}, timedelta(0)
    
    # 1. Define Resources based on Pair
    if pair_name == 'Pair A (T1&T2)':
        resources = {'T1': RATE_T1, 'T2': RATE_T2}
    else:
        resources = {'T3': RATE_T3, 'T4': RATE_T4}
        
    # 2. DYNAMIC SORT: Find which tippler is free earliest
    sorted_tipplers = sorted(resources.keys(), key=lambda x: tippler_state[x])
    
    t_primary = sorted_tipplers[0]   # The one free earliest
    t_secondary = sorted_tipplers[1] # The one free later
    
    # 3. Calculate IDLE TIME (Gap between Machine Free and Train Ready)
    machine_free_at = tippler_state[t_primary]
    
    if machine_free_at == pd.Timestamp.min:
        idle_delta = timedelta(0)
    else:
        idle_delta = max(timedelta(0), ready_time - machine_free_at)

    # 4. Split Wagons
    w_first = min(wagons, WAGONS_FIRST_BATCH)
    w_second = wagons - w_first
    
    # 5. Calculate Timings
    
    # Primary Batch
    start_primary = max(ready_time, tippler_state[t_primary])
    finish_primary = start_primary + timedelta(hours=(w_first / resources[t_primary]))
    
    updated_times = {}
    updated_times[t_primary] = finish_primary
    
    # Secondary Batch
    if w_second > 0:
        start_secondary_theoretical = ready_time + timedelta(minutes=INTER_TIPPLER_DELAY)
        start_secondary = max(start_secondary_theoretical, tippler_state[t_secondary])
        finish_secondary = start_secondary + timedelta(hours=(w_second / resources[t_secondary]))
        
        updated_times[t_secondary] = finish_secondary
        
        overall_finish = max(finish_primary, finish_secondary)
        used_str = f"{t_primary} & {t_secondary}"
    else:
        overall_finish = finish_primary
        used_str = f"{t_primary} Only"
        updated_times[t_secondary] = tippler_state[t_secondary]

    return overall_finish, used_str, start_primary, updated_times, idle_delta

def get_line_entry_time(group_name, arrival_time, line_groups):
    group = line_groups[group_name]
    active_line_blocks = sorted([t for t in group['line_free_times'] if t > arrival_time])
    if len(active_line_blocks) < group['capacity']:
        return arrival_time
    slots_needed_to_free = len(active_line_blocks) - group['capacity'] + 1
    return active_line_blocks[slots_needed_to_free - 1]

def find_specific_line(group_name, entry_time, specific_line_status, last_used_index):
    if group_name == 'Group_Lines_8_10':
        base_candidates = [8, 9, 10]
        start_idx = (last_used_index + 1) % 3
        priority_candidates = base_candidates[start_idx:] + base_candidates[:start_idx]
    else:
        priority_candidates = [11]
    for line_num in priority_candidates:
        if specific_line_status[line_num] <= entry_time:
            return line_num
    return priority_candidates[0]

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

def format_duration_hhmm(delta):
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"

# ==========================================
# 3. MAIN APP EXECUTION
# ==========================================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.upper()
    
    required_cols = ['TOTL UNTS', 'EXPD ARVLTIME']
    
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        st.error(f"Error: Missing columns {missing}")
    else:
        # FILTER: Exclude 'LDNG'
        if 'PLCT RESN' in df.columns:
            df = df[~df['PLCT RESN'].astype(str).str.upper().str.contains('LDNG', na=False)]

        df['wagon_count'] = df['TOTL UNTS'].apply(parse_wagons)
        df['exp_arrival_dt'] = pd.to_datetime(df['EXPD ARVLTIME'], errors='coerce')
        
        if 'STTS CODE' in df.columns and 'STTS TIME' in df.columns:
            df['stts_time_dt'] = pd.to_datetime(df['STTS TIME'], errors='coerce')
            def calculate_effective_arrival(row):
                if str(row.get('STTS CODE')).strip() == 'PL' and pd.notnull(row['stts_time_dt']):
                    return row['stts_time_dt']
                return row['exp_arrival_dt']
            df['arrival_dt'] = df.apply(calculate_effective_arrival, axis=1)
        else:
            df['arrival_dt'] = df['exp_arrival_dt']

        df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt').reset_index(drop=True)

        # --- STATE VARIABLES ---
        tippler_state = {
            'T1': pd.Timestamp.min, 'T2': pd.Timestamp.min,
            'T3': pd.Timestamp.min, 'T4': pd.Timestamp.min
        }
        specific_line_status = {8: pd.Timestamp.min, 9: pd.Timestamp.min, 10: pd.Timestamp.min, 11: pd.Timestamp.min}
        line_groups = {
            'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}, 
            'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}    
        }
        rr_tracker_A = -1 
        
        assignments = []
        daily_demurrage_tracker = {} # Key: Date String, Value: Total Seconds
        
        for _, rake in df.iterrows():
            
            # --- SCENARIO A: Pair A ---
            shunt_offset_A = timedelta(minutes=SHUNT_MINS_A)
            entry_time_A = get_line_entry_time('Group_Lines_8_10', rake['arrival_dt'], line_groups)
            ready_to_unload_A = entry_time_A + shunt_offset_A
            
            finish_A, used_A, start_A, updated_times_A, idle_A = calculate_split_finish(
                rake['wagon_count'], 'Pair A (T1&T2)', ready_to_unload_A, tippler_state
            )
            
            # --- SCENARIO B: Pair B ---
            shunt_offset_B = timedelta(minutes=SHUNT_MINS_B)
            entry_time_B = get_line_entry_time('Group_Line_11', rake['arrival_dt'], line_groups)
            ready_to_unload_B = entry_time_B + shunt_offset_B
            
            finish_B, used_B, start_B, updated_times_B, idle_B = calculate_split_finish(
                rake['wagon_count'], 'Pair B (T3&T4)', ready_to_unload_B, tippler_state
            )
            
            # --- DECISION ---
            if finish_A <= finish_B:
                best_pair = 'Pair A (T1&T2)'
                best_grp = 'Group_Lines_8_10'
                best_entry = entry_time_A
                best_ready = ready_to_unload_A
                best_start = start_A
                best_finish = finish_A
                best_tipplers = used_A
                best_idle = idle_A
                best_formation_mins = FORMATION_MINS_A
                
                # Update State
                for t_id, t_time in updated_times_A.items(): tippler_state[t_id] = t_time
                res_t1 = updated_times_A.get('T1', pd.NaT)
                res_t2 = updated_times_A.get('T2', pd.NaT)
                res_t3 = pd.NaT
                res_t4 = pd.NaT
                
                selected_line = find_specific_line(best_grp, best_entry, specific_line_status, rr_tracker_A)
                if selected_line in [8, 9, 10]:
                    mapping = {8:0, 9:1, 10:2}
                    rr_tracker_A = mapping[selected_line]
            else:
                best_pair = 'Pair B (T3&T4)'
                best_grp = 'Group_Line_11'
                best_entry = entry_time_B
                best_ready = ready_to_unload_B
                best_start = start_B
                best_finish = finish_B
                best_tipplers = used_B
                best_idle = idle_B
                best_formation_mins = FORMATION_MINS_B
                
                # Update State
                for t_id, t_time in updated_times_B.items(): tippler_state[t_id] = t_time
                res_t1 = pd.NaT
                res_t2 = pd.NaT
                res_t3 = updated_times_B.get('T3', pd.NaT)
                res_t4 = updated_times_B.get('T4', pd.NaT)
                
                selected_line = 11
                
            # --- UPDATE INFRASTRUCTURE ---
            clearance_mins = line_groups[best_grp]['clearance_mins']
            line_free_at = best_entry + timedelta(minutes=clearance_mins)
            line_groups[best_grp]['line_free_times'].append(line_free_at)
            specific_line_status[selected_line] = line_free_at
            
            # --- CALCULATE DURATIONS ---
            wait_delta = best_start - best_ready
            
            formation_delta = timedelta(minutes=best_formation_mins)
            total_duration_delta = (best_finish - rake['arrival_dt']) + formation_delta
            
            # --- DEMURRAGE CALCULATION ---
            # Demurrage = Max(0, Total Duration - Free Time)
            free_time_delta = timedelta(hours=FREE_TIME_HOURS)
            demurrage_delta = max(timedelta(0), total_duration_delta - free_time_delta)
            
            # Aggregate Daily Demurrage
            arrival_date_str = rake['arrival_dt'].strftime('%Y-%m-%d')
            if arrival_date_str not in daily_demurrage_tracker:
                daily_demurrage_tracker[arrival_date_str] = 0.0
            daily_demurrage_tracker[arrival_date_str] += demurrage_delta.total_seconds()
            
            assignments.append({
                'Rake': rake['RAKE NAME'],
                'Station From': rake.get('STTN FROM', 'N/A'),
                'Status': rake.get('STTS CODE', 'N/A'),
                'Wagons': rake['wagon_count'],
                'Original Arrival': format_dt(rake['exp_arrival_dt']),
                'Revised Arrival Time': format_dt(rake['arrival_dt']),
                'Line Allotted': selected_line,
                'Line Entry Time': format_dt(best_entry), 
                'Shunting Complete': format_dt(best_ready), 
                'Assigned Pair': best_pair,
                'Tipplers Used': best_tipplers,
                'Tippler Start Time': format_dt(best_start),
                'Finish Unload': format_dt(best_finish),
                'T1 Finish': format_dt(res_t1),
                'T2 Finish': format_dt(res_t2),
                'T3 Finish': format_dt(res_t3),
                'T4 Finish': format_dt(res_t4),
                'Wait (Tippler)': format_duration_hhmm(wait_delta),
                'Tippler Idle Time': format_duration_hhmm(best_idle), 
                'Formation Time': f"{int(best_formation_mins)} m",
                'Total Duration': format_duration_hhmm(total_duration_delta),
                'Demurrage': format_duration_hhmm(demurrage_delta), # NEW COLUMN
                'Placement Reason': rake.get('PLCT RESN', 'N/A')
            })

        res_df = pd.DataFrame(assignments)
        
        # --- DISPLAY MAIN TABLE ---
        st.success(f"Optimization Complete! Processed {len(res_df)} trains.")
        st.dataframe(res_df, use_container_width=True)
        
        # --- DISPLAY DAILY DEMURRAGE SUMMARY ---
        st.markdown("### 📅 Daily Demurrage Summary")
        if daily_demurrage_tracker:
            daily_data = []
            for date_str, total_seconds in daily_demurrage_tracker.items():
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                formatted_time = f"{hours:02d}:{minutes:02d}"
                daily_data.append({'Date': date_str, 'Total Demurrage Hours': formatted_time})
            
            daily_df = pd.DataFrame(daily_data).sort_values('Date')
            st.dataframe(daily_df, use_container_width=True)
        else:
            st.info("No demurrage incurred based on current settings.")
        
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Optimized Schedule", csv, "optimized_schedule.csv", "text/csv")
