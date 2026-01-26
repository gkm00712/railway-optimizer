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

# Unloading Rates
RATE_PAIR_A = st.sidebar.number_input("Pair A Rate (Wagons/hr)", value=6.0, step=0.5)
RATE_PAIR_B = st.sidebar.number_input("Pair B Rate (Wagons/hr)", value=9.0, step=0.5)

# Shunting Penalties
SHUNT_MINS_A = st.sidebar.number_input("Pair A Shunt Penalty (Mins)", value=50.0, step=5.0)
SHUNT_MINS_B = st.sidebar.number_input("Pair B Shunt Penalty (Mins)", value=100.0, step=5.0)

# Line Clearance Constraints
st.sidebar.info(f"Line 8-10 Clearance: 50 mins (Parallel Processing)\nLine 11 Clearance: 100 mins (Parallel Processing)")

# ==========================================
# 2. CALCULATION LOGIC
# ==========================================

def get_duration(wagons, pair_name):
    """Calculates total processing time for a split rake."""
    if wagons == 0: return 0
    w_per_tippler = wagons / 2.0
    
    if pair_name == 'Pair A (T1&T2)':
        unload_hrs = w_per_tippler / RATE_PAIR_A
        shunt_hrs = SHUNT_MINS_A / 60.0
    else:
        unload_hrs = w_per_tippler / RATE_PAIR_B
        shunt_hrs = SHUNT_MINS_B / 60.0
        
    return unload_hrs + shunt_hrs

def get_line_entry_time(group_name, arrival_time, line_groups):
    """
    Finds the earliest time a train can physically ENTER the line.
    This depends ONLY on the Line Clearance (50/100 mins), not the Tippler.
    """
    group = line_groups[group_name]
    
    # 1. Filter for trains that are currently blocking the lines (50/100 min timer)
    active_line_blocks = sorted([t for t in group['line_free_times'] if t > arrival_time])
    
    # 2. Capacity Check
    # If fewer lines are blocked than the allowed capacity, we can enter immediately.
    # Group A Cap = 2 (Means 2 lines busy, 1 kept vacant).
    if len(active_line_blocks) < group['capacity']:
        return arrival_time
    
    # 3. Queue Logic
    # If lines are full, we wait for the 'Nth' train's line timer to expire.
    # e.g. If Capacity is 2, and 2 trains are blocking lines, wait for the 1st one to clear.
    slots_needed_to_free = len(active_line_blocks) - group['capacity'] + 1
    return active_line_blocks[slots_needed_to_free - 1]

def find_specific_line(group_name, entry_time, specific_line_status, last_used_index):
    """
    Selects a specific line number using Round-Robin logic.
    It checks which specific line is free at 'entry_time'.
    """
    if group_name == 'Group_Lines_8_10':
        base_candidates = [8, 9, 10]
        # Rotate logic: Start checking from next index
        start_idx = (last_used_index + 1) % 3
        priority_candidates = base_candidates[start_idx:] + base_candidates[:start_idx]
    else:
        priority_candidates = [11]
        
    for line_num in priority_candidates:
        if specific_line_status[line_num] <= entry_time:
            return line_num
            
    return priority_candidates[0] # Fallback

def parse_wagons(val):
    try:
        if '+' in str(val):
            parts = str(val).split('+')
            return sum(int(p) for p in parts if p.strip().isdigit())
        return int(float(val))
    except: return 0

# ==========================================
# 3. MAIN APP EXECUTION
# ==========================================

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.upper()
    
    required_cols = ['TOTL UNTS', 'EXPD ARVLTIME']
    
    if not all(col in df.columns for col in required_cols):
        # Helper to find missing columns
        missing = [c for c in required_cols if c not in df.columns]
        st.error(f"Error: Missing columns {missing}")
    else:
        # Filter Loaded
        if 'L/E' in df.columns:
            initial_count = len(df)
            df = df[df['L/E'].astype(str).str.strip().str.upper() == 'L']
            if len(df) != initial_count:
                st.warning(f"⚠️ Filtered {initial_count - len(df)} empty trains.")
        
        # Parse Data
        df['wagon_count'] = df['TOTL UNTS'].apply(parse_wagons)
        df['exp_arrival_dt'] = pd.to_datetime(df['EXPD ARVLTIME'], errors='coerce')
        
        # Revised Arrival Logic
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
        
        # 1. Tippler State (When is the UNLOADER free?)
        pair_state = {
            'Pair A (T1&T2)': pd.Timestamp.min, 
            'Pair B (T3&T4)': pd.Timestamp.min 
        }
        
        # 2. Specific Line State (When is Line X free?)
        specific_line_status = {8: pd.Timestamp.min, 9: pd.Timestamp.min, 10: pd.Timestamp.min, 11: pd.Timestamp.min}
        
        # 3. Group Capacity State (To track the "50 min" slots)
        line_groups = {
            'Group_Lines_8_10': {
                'capacity': 2, 
                'clearance_mins': 50, 
                'line_free_times': [] # Stores when the *LINE* becomes free (Entry + 50)
            }, 
            'Group_Line_11': {
                'capacity': 1, 
                'clearance_mins': 100, 
                'line_free_times': [] 
            }    
        }
        
        rr_tracker_A = -1 
        
        assignments = []
        
        for _, rake in df.iterrows():
            
            # --- SCENARIO A: Pair A ---
            dur_A = get_duration(rake['wagon_count'], 'Pair A (T1&T2)')
            grp_A = 'Group_Lines_8_10'
            
            # Step 1: When can we enter the line? (Decoupled from Tippler)
            entry_time_A = get_line_entry_time(grp_A, rake['arrival_dt'], line_groups)
            
            # Step 2: When does unloading start? (Max of Entry OR Tippler Free)
            start_unload_A = max(entry_time_A, pair_state['Pair A (T1&T2)'])
            finish_A = start_unload_A + timedelta(hours=dur_A)
            
            # --- SCENARIO B: Pair B ---
            dur_B = get_duration(rake['wagon_count'], 'Pair B (T3&T4)')
            grp_B = 'Group_Line_11'
            
            entry_time_B = get_line_entry_time(grp_B, rake['arrival_dt'], line_groups)
            start_unload_B = max(entry_time_B, pair_state['Pair B (T3&T4)'])
            finish_B = start_unload_B + timedelta(hours=dur_B)
            
            # --- DECISION ---
            if finish_A <= finish_B:
                best_pair = 'Pair A (T1&T2)'
                best_grp = grp_A
                best_entry = entry_time_A
                best_start = start_unload_A
                best_finish = finish_A
                best_dur = dur_A
                
                # Update Tippler
                pair_state['Pair A (T1&T2)'] = finish_A
                
                # Assign Line
                selected_line = find_specific_line(grp_A, best_entry, specific_line_status, rr_tracker_A)
                if selected_line in [8, 9, 10]:
                    mapping = {8:0, 9:1, 10:2}
                    rr_tracker_A = mapping[selected_line]
            else:
                best_pair = 'Pair B (T3&T4)'
                best_grp = grp_B
                best_entry = entry_time_B
                best_start = start_unload_B
                best_finish = finish_B
                best_dur = dur_B
                
                pair_state['Pair B (T3&T4)'] = finish_B
                selected_line = 11
                
            # --- UPDATE INFRASTRUCTURE ---
            # IMPORTANT: The Line is blocked from Entry Time -> Entry Time + Clearance
            # It does NOT wait for unloading to finish.
            clearance_mins = line_groups[best_grp]['clearance_mins']
            line_free_at = best_entry + timedelta(minutes=clearance_mins)
            
            # Update Group State (for capacity check)
            line_groups[best_grp]['line_free_times'].append(line_free_at)
            
            # Update Specific Line State
            specific_line_status[selected_line] = line_free_at
            
            wait_mins = (best_start - rake['arrival_dt']).total_seconds() / 60
            
            assignments.append({
                'Rake': rake['RAKE NAME'],
                'Station From': rake.get('STTN FROM', 'N/A'),
                'Status': rake.get('STTS CODE', 'N/A'),
                'Wagons': rake['wagon_count'],
                'Original Arrival': rake['exp_arrival_dt'].strftime('%d-%H:%M'),
                'Revised Arrival Time': rake['arrival_dt'].strftime('%d-%H:%M'),
                'Line Allotted': selected_line,
                'Line Entry Time': best_entry.strftime('%d-%H:%M'), # NEW COLUMN
                'Assigned': best_pair,
                'Duration': f"{best_dur:.2f}h",
                'Start Unload': best_start.strftime('%d-%H:%M'),
                'Finish Unload': best_finish.strftime('%d-%H:%M'),
                'Wait': f"{int(wait_mins)} m",
                'Placement Reason': rake.get('PLCT RESN', 'N/A')
            })

        res_df = pd.DataFrame(assignments)
        st.success(f"Optimization Complete! Processed {len(res_df)} loaded trains.")
        st.dataframe(res_df, use_container_width=True)
        
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Optimized Schedule", csv, "optimized_schedule.csv", "text/csv")
