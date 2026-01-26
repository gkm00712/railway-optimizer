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
    
    # Split Rake: Half wagons go to each tippler
    w_per_tippler = wagons / 2.0
    
    if pair_name == 'Pair A (T1&T2)':
        unload_hrs = w_per_tippler / RATE_PAIR_A
        shunt_hrs = SHUNT_MINS_A / 60.0
    else:
        unload_hrs = w_per_tippler / RATE_PAIR_B
        shunt_hrs = SHUNT_MINS_B / 60.0
        
    return unload_hrs + shunt_hrs

def get_line_entry_time(group_name, arrival_time, line_groups):
    """Finds the earliest time a train can enter the line group."""
    group = line_groups[group_name]
    active = sorted([t for t in group['active_until'] if t > arrival_time])
    
    if len(active) < group['capacity']:
        return arrival_time
    
    return min(active)

def parse_wagons(val):
    """Handles standard integers and '58+1' wagon formats."""
    try:
        if '+' in str(val):
            parts = str(val).split('+')
            return sum(int(p) for p in parts if p.strip().isdigit())
        return int(float(val))
    except: return 0

# ==========================================
# 3. MAIN APP EXECUTION
# ==========================================

# File Uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # --- DATA PARSING & CLEANING ---
    # Required columns check
    required_cols = ['TOTL UNTS', 'EXPD ARVLTIME']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: CSV must contain {required_cols} columns.")
    else:
        # 1. Parse Wagon Counts
        df['wagon_count'] = df['TOTL UNTS'].apply(parse_wagons)
        
        # 2. Parse Timestamp Columns
        df['exp_arrival_dt'] = pd.to_datetime(df['EXPD ARVLTIME'], errors='coerce')
        
        # Check if STTS columns exist for the new logic
        if 'STTS' in df.columns and 'STTS TIME' in df.columns:
            df['stts_time_dt'] = pd.to_datetime(df['STTS TIME'], errors='coerce')
            
            # --- NEW REQUIREMENT LOGIC START ---
            def calculate_effective_arrival(row):
                # If Status is PL (Placed) and STTS TIME is valid, use STTS TIME
                if str(row['STTS']).strip().upper() == 'PL' and pd.notnull(row['stts_time_dt']):
                    return row['stts_time_dt']
                # Otherwise, default to Expected Arrival Time
                return row['exp_arrival_dt']
            
            df['arrival_dt'] = df.apply(calculate_effective_arrival, axis=1)
            # --- NEW REQUIREMENT LOGIC END ---
        else:
            # Fallback if STTS columns are missing in CSV
            st.warning("Columns 'STTS' or 'STTS TIME' not found. Defaulting to 'EXPD ARVLTIME'.")
            df['arrival_dt'] = df['exp_arrival_dt']

        # 3. Sort and Drop Invalid Rows
        df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt').reset_index(drop=True)

        # Initialize State Variables
        pair_state = {
            'Pair A (T1&T2)': pd.Timestamp.min, 
            'Pair B (T3&T4)': pd.Timestamp.min 
        }
        
        # Reset Line Constraints
        line_groups = {
            'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 25, 'active_until': []}, 
            'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'active_until': []}    
        }
        
        assignments = []
        
        # Run Simulation Loop
        for _, rake in df.iterrows():
            
            # --- Option 1: Pair A (T1 & T2) ---
            dur_A = get_duration(rake['wagon_count'], 'Pair A (T1&T2)')
            grp_A = 'Group_Lines_8_10'
            line_free_A = get_line_entry_time(grp_A, rake['arrival_dt'], line_groups)
            start_A = max(rake['arrival_dt'], line_free_A, pair_state['Pair A (T1&T2)'])
            finish_A = start_A + timedelta(hours=dur_A)
            
            # --- Option 2: Pair B (T3 & T4) ---
            dur_B = get_duration(rake['wagon_count'], 'Pair B (T3&T4)')
            grp_B = 'Group_Line_11'
            line_free_B = get_line_entry_time(grp_B, rake['arrival_dt'], line_groups)
            start_B = max(rake['arrival_dt'], line_free_B, pair_state['Pair B (T3&T4)'])
            finish_B = start_B + timedelta(hours=dur_B)
            
            # --- Decision: Earliest Finish ---
            finish_diff = abs((finish_A - finish_B).total_seconds() / 60)
            
            if finish_A <= finish_B:
                best_pair = 'Pair A (T1&T2)'
                best_grp = grp_A
                best_start = start_A
                best_finish = finish_A
                best_dur = dur_A
                pair_state['Pair A (T1&T2)'] = finish_A
                
                reason = f"Finishes {int(finish_diff)}m earlier than Pair B (would finish {finish_B.strftime('%d-%H:%M')})"
            else:
                best_pair = 'Pair B (T3&T4)'
                best_grp = grp_B
                best_start = start_B
                best_finish = finish_B
                best_dur = dur_B
                pair_state['Pair B (T3&T4)'] = finish_B
                
                reason = f"Finishes {int(finish_diff)}m earlier than Pair A (would finish {finish_A.strftime('%d-%H:%M')})"
                
            # Update Line Clearance
            clearance = line_groups[best_grp]['clearance_mins']
            line_groups[best_grp]['active_until'].append(best_start + timedelta(minutes=clearance))
            
            # --- CAPTURE EXTRA COLUMNS ---
            # 1. Placement Reason
            placement_reason = rake['PLCT RESN'] if 'PLCT RESN' in df.columns else 'N/A'
            # 2. Station From
            station_from = rake['STTN FROM'] if 'STTN FROM' in df.columns else 'N/A'
            # 3. Status (Capture for debugging/viewing)
            status_code = rake['STTS'] if 'STTS' in df.columns else 'N/A'

            # Log Result
            wait_mins = (best_start - rake['arrival_dt']).total_seconds() / 60
            assignments.append({
                'Rake': rake['RAKE NAME'],
                'Station From': station_from,
                'Status': status_code, # Added to view status in table
                'Wagons': rake['wagon_count'],
                'Arrival': rake['arrival_dt'].strftime('%d-%H:%M'),
                'Assigned': best_pair,
                'Duration': f"{best_dur:.2f}h",
                'Start': best_start.strftime('%d-%H:%M'),
                'Finish': best_finish.strftime('%d-%H:%M'),
                'Wait': f"{int(wait_mins)} m",
                'Placement Reason': placement_reason
            })

        # --- OUTPUT RESULTS ---
        res_df = pd.DataFrame(assignments)
        
        st.success(f"Optimization Complete! Processed {len(res_df)} trains.")
        
        # Display Dataframe
        st.dataframe(res_df, use_container_width=True)
        
        # Download Button
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Optimized Schedule",
            data=csv,
            file_name="optimized_schedule.csv",
            mime="text/csv",
        )
