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

# Unloading Rates (Per Individual Tippler)
st.sidebar.subheader("Individual Tippler Rates")
RATE_T1 = st.sidebar.number_input("Tippler 1 Rate (Wagons/hr)", value=6.0, step=0.5)
RATE_T2 = st.sidebar.number_input("Tippler 2 Rate (Wagons/hr)", value=6.0, step=0.5)
RATE_T3 = st.sidebar.number_input("Tippler 3 Rate (Wagons/hr)", value=9.0, step=0.5)
RATE_T4 = st.sidebar.number_input("Tippler 4 Rate (Wagons/hr)", value=9.0, step=0.5)

# Shunting Penalties
st.sidebar.subheader("Shunting Delays")
SHUNT_MINS_A = st.sidebar.number_input("Pair A Initial Shunt (Mins)", value=50.0, step=5.0)
SHUNT_MINS_B = st.sidebar.number_input("Pair B Initial Shunt (Mins)", value=50.0, step=5.0)

# Stagger Logic
st.sidebar.subheader("Split Logic")
WAGONS_FIRST_BATCH = st.sidebar.number_input("Wagons in 1st Batch", value=30, step=1)
# Time taken to shunt the first batch so second batch can start?
# Assuming proportional shunting time or fixed delay? 
# User said: "unloading can be started at one tippler of pair after shunting time of 30 wagons"
# Let's verify if they mean the 2nd tippler starts after the 1st batch is *shunted*?
# For now, we assume a fixed delay between T1 start and T2 start equal to the shunting time of the 1st batch.
# Let's approximate Shunt Time per Wagon roughly or use a fixed "Inter-Tippler Delay".
INTER_TIPPLER_DELAY = st.sidebar.number_input("Delay for 2nd Tippler Start (Mins)", value=15.0, step=5.0)

# Line Clearance Constraints
st.sidebar.info(f"Line 8-10 Clearance: 50 mins (Parallel Processing)\nLine 11 Clearance: 100 mins (Parallel Processing)")

# ==========================================
# 2. CALCULATION LOGIC
# ==========================================

def calculate_split_finish(wagons, pair_name, start_time, tippler_state):
    """
    Calculates the finish time for a train split across two tipplers.
    Returns: (Finish Time, String of Tipplers Used)
    """
    if wagons == 0: return start_time, "None"
    
    # 1. Split Wagons
    w_first = min(wagons, WAGONS_FIRST_BATCH)
    w_second = wagons - w_first
    
    # 2. Identify Tipplers & Rates
    if pair_name == 'Pair A (T1&T2)':
        t_a, t_b = 'T1', 'T2'
        rate_a, rate_b = RATE_T1, RATE_T2
    else:
        t_a, t_b = 'T3', 'T4'
        rate_a, rate_b = RATE_T3, RATE_T4
        
    # 3. Determine Start Times for Individual Tipplers
    # First batch starts at 'start_time' (assuming tippler is free)
    # Second batch starts at 'start_time + INTER_TIPPLER_DELAY'
    
    # Check availability of Tippler A
    ready_a = max(start_time, tippler_state[t_a])
    finish_a = ready_a + timedelta(hours=(w_first / rate_a))
    
    # Check availability of Tippler B
    # It can start ONLY after the delay AND when it is free
    start_time_b_theoretical = start_time + timedelta(minutes=INTER_TIPPLER_DELAY)
    ready_b = max(start_time_b_theoretical, tippler_state[t_b])
    
    if w_second > 0:
        finish_b = ready_b + timedelta(hours=(w_second / rate_b))
        return max(finish_a, finish_b), f"{t_a} & {t_b}", finish_a, finish_b
    else:
        return finish_a, f"{t_a} Only", finish_a, ready_b # Only used first tippler

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
    if pd.isnull(dt): return "N/A"
    return dt.strftime('%d-%H:%M')

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
        # Track 4 individual tipplers now
        tippler_state = {
            'T1': pd.Timestamp.min,
            'T2': pd.Timestamp.min,
            'T3': pd.Timestamp.min,
            'T4': pd.Timestamp.min
        }
        
        specific_line_status = {8: pd.Timestamp.min, 9: pd.Timestamp.min, 10: pd.Timestamp.min, 11: pd.Timestamp.min}
        
        line_groups = {
            'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}, 
            'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}    
        }
        
        rr_tracker_A = -1 
        
        assignments = []
        
        for _, rake in df.iterrows():
            
            # --- SCENARIO A: Pair A (Lines 8-10) ---
            shunt_offset_A = timedelta(minutes=SHUNT_MINS_A)
            grp_A = 'Group_Lines_8_10'
            
            # 1. Line Entry
            entry_time_A = get_line_entry_time(grp_A, rake['arrival_dt'], line_groups)
            ready_to_unload_A = entry_time_A + shunt_offset_A
            
            # 2. Calculate Finish if using T1 & T2
            finish_A, used_A, finish_t1, finish_t2 = calculate_split_finish(
                rake['wagon_count'], 'Pair A (T1&T2)', ready_to_unload_A, tippler_state
            )
            
            # --- SCENARIO B: Pair B (Line 11) ---
            shunt_offset_B = timedelta(minutes=SHUNT_MINS_B)
            grp_B = 'Group_Line_11'
            
            # 1. Line Entry
            entry_time_B = get_line_entry_time(grp_B, rake['arrival_dt'], line_groups)
            ready_to_unload_B = entry_time_B + shunt_offset_B
            
            # 2. Calculate Finish if using T3 & T4
            finish_B, used_B, finish_t3, finish_t4 = calculate_split_finish(
                rake['wagon_count'], 'Pair B (T3&T4)', ready_to_unload_B, tippler_state
            )
            
            # --- DECISION ---
            if finish_A <= finish_B:
                best_pair = 'Pair A (T1&T2)'
                best_grp = grp_A
                best_entry = entry_time_A
                best_ready = ready_to_unload_A
                # Actual start is the earliest start of the first batch
                best_start = max(ready_to_unload_A, tippler_state['T1']) 
                best_finish = finish_A
                best_tipplers = used_A
                
                # Update Individual Tipplers
                if 'T1' in used_A: tippler_state['T1'] = finish_t1
                if 'T2' in used_A: tippler_state['T2'] = finish_t2
                
                selected_line = find_specific_line(grp_A, best_entry, specific_line_status, rr_tracker_A)
                if selected_line in [8, 9, 10]:
                    mapping = {8:0, 9:1, 10:2}
                    rr_tracker_A = mapping[selected_line]
            else:
                best_pair = 'Pair B (T3&T4)'
                best_grp = grp_B
                best_entry = entry_time_B
                best_ready = ready_to_unload_B
                best_start = max(ready_to_unload_B, tippler_state['T3'])
                best_finish = finish_B
                best_tipplers = used_B
                
                # Update Individual Tipplers
                if 'T3' in used_B: tippler_state['T3'] = finish_t3
                if 'T4' in used_B: tippler_state['T4'] = finish_t4
                
                selected_line = 11
                
            # --- UPDATE INFRASTRUCTURE ---
            clearance_mins = line_groups[best_grp]['clearance_mins']
            line_free_at = best_entry + timedelta(minutes=clearance_mins)
            
            line_groups[best_grp]['line_free_times'].append(line_free_at)
            specific_line_status[selected_line] = line_free_at
            
            wait_mins = (best_start - rake['arrival_dt']).total_seconds() / 60
            dur_mins = (best_finish - best_start).total_seconds() / 60.0
            
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
                'Tipplers Used': best_tipplers, # NEW COLUMN
                'Actual Start Unload': format_dt(best_start),
                'Finish Unload': format_dt(best_finish),
                'Wait': f"{int(wait_mins)} m",
                'Duration (Hrs)': f"{dur_mins/60.0:.2f}h",
                'Placement Reason': rake.get('PLCT RESN', 'N/A')
            })

        res_df = pd.DataFrame(assignments)
        st.success(f"Optimization Complete! Processed {len(res_df)} trains.")
        st.dataframe(res_df, use_container_width=True)
        
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Optimized Schedule", csv, "optimized_schedule.csv", "text/csv")
