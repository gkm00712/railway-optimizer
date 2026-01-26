import streamlit as st
import pandas as pd
from datetime import timedelta
import math

# --- PAGE CONFIGURATION ---
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
st.sidebar.info(f"Line 8-10 Clearance: 25 mins\nLine 11 Clearance: 100 mins")

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
    Finds the earliest time a train can enter the group based on CAPACITY.
    Group 8-10: Capacity 2 (Ensures 1 out of 3 is always vacant).
    Group 11: Capacity 1.
    """
    group = line_groups[group_name]
    # Filter for active trains that finish AFTER arrival
    active_busy_times = sorted([t for t in group['active_slots'] if t > arrival_time])
    
    # If currently active trains < capacity, we can start immediately
    if len(active_busy_times) < group['capacity']:
        return arrival_time
    
    # Otherwise, wait for the earliest slot to open
    # We take the Nth earliest finish time where N is capacity
    # e.g., if cap=2 and 2 trains are busy, wait for min(finish_times)
    
    # Logic: The (len - capacity)th element is the one we wait for? 
    # Actually simpler: If cap is 2, and we have 2 busy, we wait for the 1st one to finish.
    # If cap is 2 and we have 3 busy (shouldn't happen), we wait.
    
    slots_needed_to_free = len(active_busy_times) - group['capacity'] + 1
    # We need to wait until the 'slots_needed_to_free'-th train leaves.
    # Since active_busy_times is sorted, we pick index [slots_needed_to_free - 1]
    
    return active_busy_times[slots_needed_to_free - 1]

def find_specific_line(group_name, start_time, specific_line_status):
    """
    Determines exactly which line number (8, 9, 10, or 11) is used.
    """
    if group_name == 'Group_Lines_8_10':
        candidates = [8, 9, 10]
    else:
        candidates = [11]
        
    # Pick the first candidate that is free at or before start_time
    for line_num in candidates:
        if specific_line_status[line_num] <= start_time:
            return line_num
            
    # Fallback (Should not happen if capacity logic is correct)
    return candidates[0] 

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
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.upper()
    
    required_cols = ['TOTL UNTS', 'EXPD ARVLTIME']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: Missing columns. Found: {df.columns.tolist()}")
    else:
        # Filter Loaded
        if 'L/E' in df.columns:
            df = df[df['L/E'].astype(str).str.strip().str.upper() == 'L']
        
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
        pair_state = {
            'Pair A (T1&T2)': pd.Timestamp.min, 
            'Pair B (T3&T4)': pd.Timestamp.min 
        }
        
        # Track when specific lines (8,9,10,11) become free
        specific_line_status = {
            8: pd.Timestamp.min,
            9: pd.Timestamp.min,
            10: pd.Timestamp.min,
            11: pd.Timestamp.min
        }

        # Track "Slots" for capacity logic
        line_groups = {
            'Group_Lines_8_10': {
                'capacity': 2, # Limits usage to 2 lines, keeping 1 vacant
                'clearance_mins': 25, 
                'active_slots': [] 
            }, 
            'Group_Line_11': {
                'capacity': 1, 
                'clearance_mins': 100, 
                'active_slots': []
            }    
        }
        
        assignments = []
        
        for _, rake in df.iterrows():
            
            # --- Option A: Pair A (Lines 8,9,10) ---
            dur_A = get_duration(rake['wagon_count'], 'Pair A (T1&T2)')
            grp_A = 'Group_Lines_8_10'
            
            # 1. When can we physically start based on group capacity?
            line_free_A = get_line_entry_time(grp_A, rake['arrival_dt'], line_groups)
            # 2. When is the Tippler Pair free?
            pair_free_A = pair_state['Pair A (T1&T2)']
            
            start_A = max(rake['arrival_dt'], line_free_A, pair_free_A)
            finish_A = start_A + timedelta(hours=dur_A)
            
            # --- Option B: Pair B (Line 11) ---
            dur_B = get_duration(rake['wagon_count'], 'Pair B (T3&T4)')
            grp_B = 'Group_Line_11'
            
            line_free_B = get_line_entry_time(grp_B, rake['arrival_dt'], line_groups)
            pair_free_B = pair_state['Pair B (T3&T4)']
            
            start_B = max(rake['arrival_dt'], line_free_B, pair_free_B)
            finish_B = start_B + timedelta(hours=dur_B)
            
            # --- DECISION ---
            if finish_A <= finish_B:
                # Winner: PAIR A
                best_pair = 'Pair A (T1&T2)'
                best_grp = grp_A
                best_start = start_A
                best_finish = finish_A
                best_dur = dur_A
                
                # Update Tippler State
                pair_state['Pair A (T1&T2)'] = finish_A
                
                # Assign Specific Line (8, 9, or 10)
                selected_line = find_specific_line(grp_A, best_start, specific_line_status)
                
            else:
                # Winner: PAIR B
                best_pair = 'Pair B (T3&T4)'
                best_grp = grp_B
                best_start = start_B
                best_finish = finish_B
                best_dur = dur_B
                
                # Update Tippler State
                pair_state['Pair B (T3&T4)'] = finish_B
                
                # Assign Specific Line (11)
                selected_line = 11

            # --- UPDATE INFRASTRUCTURE STATE ---
            clearance = line_groups[best_grp]['clearance_mins']
            block_until = best_start + timedelta(minutes=clearance)
            
            # 1. Update Group Capacity Slots (used for next train's timing calculation)
            line_groups[best_grp]['active_slots'].append(block_until)
            
            # 2. Update Specific Line Status (used for naming the line)
            # Note: The line is technically occupied by the train until it leaves the line.
            # However, for 'placement' logic, we block it for the clearance duration?
            # Actually, usually line is blocked until Unloading + Shunting is done?
            # For this logic, we align specific line busy time with the "Slot" busy time.
            specific_line_status[selected_line] = block_until
            
            # Log
            wait_mins = (best_start - rake['arrival_dt']).total_seconds() / 60
            assignments.append({
                'Rake': rake['RAKE NAME'],
                'Station From': rake.get('STTN FROM', 'N/A'),
                'Status': rake.get('STTS CODE', 'N/A'),
                'Wagons': rake['wagon_count'],
                'Original Arrival': rake['exp_arrival_dt'].strftime('%d-%H:%M'),
                'Revised Arrival Time': rake['arrival_dt'].strftime('%d-%H:%M'),
                'Line Allotted': str(selected_line), # Shows 8, 9, 10, or 11
                'Assigned': best_pair,
                'Duration': f"{best_dur:.2f}h",
                'Start': best_start.strftime('%d-%H:%M'),
                'Finish': best_finish.strftime('%d-%H:%M'),
                'Wait': f"{int(wait_mins)} m",
                'Placement Reason': rake.get('PLCT RESN', 'N/A')
            })

        res_df = pd.DataFrame(assignments)
        st.success(f"Optimization Complete! Processed {len(res_df)} loaded trains.")
        st.dataframe(res_df, use_container_width=True)
        
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Optimized Schedule", csv, "optimized_schedule.csv", "text/csv")
