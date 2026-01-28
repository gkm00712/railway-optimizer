import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import pytz
import math

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer (Final)", layout="wide")
st.title("🚂 Advanced Rake Optimization: Dynamic Breakdown & Rerouting")
st.markdown("""
**System Status:** `IST Timezone Active` | **Logic:** `Recursive Breakdown Handling`
- **Dynamic Failover:** If a tippler fails, wagons are automatically rerouted.
- **Priority Logic:** Checks **Partner Tippler** (T1↔T2) vs **Cross-Transfer** (T1↔T3) based on shunting penalties.
- **Constraints:** Enforces 10-min parallel delay + specific shunting times.
""")

IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def to_ist(dt):
    if pd.isnull(dt): return pd.NaT
    if dt.tzinfo is None: return IST.localize(dt)
    return dt.astimezone(IST)

def format_dt(dt):
    if pd.isnull(dt): return ""
    if dt.tzinfo is None: dt = IST.localize(dt)
    return dt.strftime('%d-%H:%M')

def format_hhmm(delta):
    if pd.isnull(delta): return ""
    total_sec = int(delta.total_seconds())
    sign = "-" if total_sec < 0 else ""
    total_sec = abs(total_sec)
    hh, mm = divmod(total_sec // 60, 60)
    return f"{sign}{hh:02d}:{mm:02d}"

def parse_wagons(val):
    try:
        if '+' in str(val):
            return sum(int(p) for p in str(val).split('+') if p.strip().isdigit())
        return int(float(val))
    except: return 0

def restore_dt(dt_str, ref_dt):
    """Restores datetime from 'dd-HH:MM' string."""
    if not isinstance(dt_str, str) or not dt_str.strip(): return pd.NaT
    try:
        parts = dt_str.split('-')
        d, t = int(parts[0]), parts[1].split(':')
        new_dt = ref_dt.replace(day=d, hour=int(t[0]), minute=int(t[1]), second=0)
        # Handle month boundary
        if d < ref_dt.day - 15: new_dt += pd.DateOffset(months=1)
        elif d > ref_dt.day + 15: new_dt -= pd.DateOffset(months=1)
        return new_dt
    except: return pd.NaT

# ==========================================
# 3. ADVANCED SIMULATION ENGINE
# ==========================================

def get_breakdown_event(tippler, start, end, downtime_list):
    """Returns the START time of a breakdown if it overlaps with the window."""
    if start.tzinfo is None: start = IST.localize(start)
    if end.tzinfo is None: end = IST.localize(end)
    for dt in downtime_list:
        if dt['Tippler'] == tippler:
            # Breakdown starts strictly inside the operation
            if start < dt['Start'] < end:
                return dt['Start'], dt['End']
    return None, None

def solve_unloading_path(wagons, preferred_tippler, ready_time, tippler_state, downtimes, rates, params, depth=0):
    """
    Recursive function to solve unloading. 
    If breakdown -> Recursively solves for remaining wagons on best available tippler.
    """
    # BASE CASE: Work Finished
    if wagons <= 0: 
        return ready_time, [], tippler_state, {}
        
    if depth > 3: # Avoid infinite loops
        return ready_time + timedelta(hours=24), ["Max Depth Exceeded"], tippler_state, {}

    # 1. Determine Actual Start on Preferred Tippler
    # Safety Check for Key Existence
    if preferred_tippler not in tippler_state:
        free_at = IST.localize(datetime(2000, 1, 1))
    else:
        free_at = tippler_state[preferred_tippler]

    if free_at.tzinfo is None: free_at = IST.localize(free_at)
    
    # 10 min setup delay if this is a transfer (depth > 0)
    setup_delay = timedelta(minutes=10) if depth > 0 else timedelta(0)
    effective_ready = ready_time + setup_delay
    
    start_time = max(effective_ready, free_at)

    # 2. Check if Start Time lands IN a downtime (Push forward logic)
    changed = True
    while changed:
        changed = False
        for dt in downtimes:
            if dt['Tippler'] == preferred_tippler and dt['Start'] <= start_time < dt['End']:
                start_time = dt['End']
                changed = True

    # 3. Calculate Theoretical Finish
    rate = rates[preferred_tippler]
    duration = timedelta(hours=wagons / rate)
    finish_time = start_time + duration

    # 4. Check for Mid-Operation Breakdown
    bk_start, bk_end = get_breakdown_event(preferred_tippler, start_time, finish_time, downtimes)

    timings = {} 
    
    # Calculate Idle for this specific segment
    idle_val = max(timedelta(0), start_time - free_at)
    timings[f"{preferred_tippler}_Idle_{depth}"] = idle_val 

    if not bk_start:
        # --- SUCCESS CASE ---
        # Update State
        timings[f"{preferred_tippler}_Start"] = start_time
        timings[f"{preferred_tippler}_End"] = finish_time
        
        # Update the state object (copy passed in recursion, so we update purely for return)
        tippler_state[preferred_tippler] = finish_time
        return finish_time, [preferred_tippler], tippler_state, timings

    else:
        # --- BREAKDOWN CASE (The Split) ---
        worked_duration = bk_start - start_time
        wagons_done = math.floor((worked_duration.total_seconds() / 3600) * rate)
        wagons_rem = wagons - wagons_done
        
        # Log partial work
        timings[f"{preferred_tippler}_Start"] = start_time
        timings[f"{preferred_tippler}_End"] = bk_start 
        
        candidates = []
        
        # OPTION 1: Partner Transfer (Intra Shunt)
        pairs = {'T1': 'T2', 'T2': 'T1', 'T3': 'T4', 'T4': 'T3'}
        partner = pairs[preferred_tippler]
        ready_partner = bk_start + timedelta(minutes=params['shunt_intra'])
        
        p_fin, p_used, p_state, p_tim = solve_unloading_path(
            wagons_rem, partner, ready_partner, tippler_state.copy(), downtimes, rates, params, depth+1
        )
        candidates.append({'type': 'Partner', 'fin': p_fin, 'used': [preferred_tippler] + p_used, 'state': p_state, 'tim': p_tim})

        # OPTION 2: Cross Transfer
        cross_map = {'T1': 'T3', 'T2': 'T4', 'T3': 'T1', 'T4': 'T2'}
        cross_tip = cross_map[preferred_tippler]
        ready_cross = bk_start + timedelta(minutes=params['shunt_cross'])
        
        c_fin, c_used, c_state, c_tim = solve_unloading_path(
            wagons_rem, cross_tip, ready_cross, tippler_state.copy(), downtimes, rates, params, depth+1
        )
        candidates.append({'type': 'Cross', 'fin': c_fin, 'used': [preferred_tippler] + c_used, 'state': c_state, 'tim': c_tim})

        # OPTION 3: Wait for Repair
        ready_wait = bk_end
        w_fin, w_used, w_state, w_tim = solve_unloading_path(
            wagons_rem, preferred_tippler, ready_wait, tippler_state.copy(), downtimes, rates, params, depth+1
        )
        candidates.append({'type': 'Wait', 'fin': w_fin, 'used': [preferred_tippler], 'state': w_state, 'tim': w_tim})

        # Select Winner
        best = sorted(candidates, key=lambda x: x['fin'])[0]
        
        # Merge Timings
        timings.update(best['tim'])
        
        final_state = best['state']
        # Ensure the broken tippler's state reflects the repair completion if it wasn't the chosen path
        current_finish_estimate = final_state.get(preferred_tippler, IST.localize(datetime(2000,1,1)))
        final_state[preferred_tippler] = max(current_finish_estimate, bk_end)
        
        return best['fin'], best['used'], final_state, timings

def run_simulation(df, params):
    rates = {'T1': params['rt1'], 'T2': params['rt2'], 'T3': params['rt3'], 'T4': params['rt4']}
    
    # Init Global State
    tippler_state = {k: params['start_time'] for k in ['T1', 'T2', 'T3', 'T4']}
    
    # Line Queues
    # Group A: Lines 8, 9, 10 (Capacity 2)
    # Group B: Line 11 (Capacity 1)
    line_queues = {
        'A': {'cap': 2, 'free': [], 'clear_time': 50},
        'B': {'cap': 1, 'free': [], 'clear_time': 100}
    }

    results = []

    df = df.sort_values('arrival_dt')

    for _, row in df.iterrows():
        arrival = row['arrival_dt']
        wagons = row['wagon_count']

        # --- EVALUATE STRATEGIES ---
        strategies = []

        # Strategy 1: Natural Route A (Line 8-10 -> T1/T2)
        q_a = sorted([t for t in line_queues['A']['free'] if t > arrival])
        entry_a = arrival if len(q_a) < line_queues['A']['cap'] else q_a[-line_queues['A']['cap']]
        ready_a = entry_a + timedelta(minutes=params['sa'])
        
        for start_node in ['T1', 'T2']:
            fin, used, state, tim = solve_unloading_path(
                wagons, start_node, ready_a, tippler_state.copy(), params['downtimes'], rates, params
            )
            dur = (fin - arrival) + timedelta(minutes=params['fa'])
            dem = max(timedelta(0), dur - timedelta(hours=params['ft']))
            strategies.append({
                'id': f"Route A ({start_node})", 'grp': 'A', 'entry': entry_a, 'ready': ready_a,
                'fin': fin, 'dem': dem, 'used': used, 'state': state, 'tim': tim, 'form': params['fa']
            })

        # Strategy 2: Natural Route B (Line 11 -> T3/T4)
        q_b = sorted([t for t in line_queues['B']['free'] if t > arrival])
        entry_b = arrival if len(q_b) < line_queues['B']['cap'] else q_b[-line_queues['B']['cap']]
        ready_b = entry_b + timedelta(minutes=params['sb'])
        
        for start_node in ['T3', 'T4']:
            fin, used, state, tim = solve_unloading_path(
                wagons, start_node, ready_b, tippler_state.copy(), params['downtimes'], rates, params
            )
            dur = (fin - arrival) + timedelta(minutes=params['fb'])
            dem = max(timedelta(0), dur - timedelta(hours=params['ft']))
            strategies.append({
                'id': f"Route B ({start_node})", 'grp': 'B', 'entry': entry_b, 'ready': ready_b,
                'fin': fin, 'dem': dem, 'used': used, 'state': state, 'tim': tim, 'form': params['fb']
            })

        # --- PICK WINNER ---
        # Priority: Min Demurrage > Earliest Finish
        best = sorted(strategies, key=lambda x: (x['dem'], x['fin']))[0]

        # --- COMMIT ---
        tippler_state = best['state']
        
        # Update Line State
        clr_time = best['entry'] + timedelta(minutes=line_queues[best['grp']]['clear_time'])
        line_queues[best['grp']]['free'].append(clr_time)
        
        # Flatten timings for DataFrame
        flat_timings = {}
        for t in ['T1', 'T2', 'T3', 'T4']:
            starts = [v for k,v in best['tim'].items() if k.startswith(f"{t}_Start")]
            ends = [v for k,v in best['tim'].items() if k.startswith(f"{t}_End")]
            idles = [v for k,v in best['tim'].items() if k.startswith(f"{t}_Idle")]
            
            flat_timings[f"{t} Start"] = format_dt(min(starts)) if starts else ""
            flat_timings[f"{t} End"] = format_dt(max(ends)) if ends else ""
            
            # Sum idle times if multiple segments
            tot_idle = sum(idles, timedelta(0))
            flat_timings[f"{t} Idle"] = format_hhmm(tot_idle) if starts else ""

        # Used String
        used_str = ", ".join(sorted(list(set(best['used']))))
        
        row_res = {
            'Rake': row['RAKE NAME'],
            'Wagons': wagons,
            'Load Type': row['LOAD TYPE'],
            '_Arrival_DT': arrival,
            '_Ready_DT': best['ready'],
            '_Form_Mins': best['form'],
            'Strategy': best['id'],
            'Arrival': format_dt(arrival),
            'Placement': format_dt(best['ready']),
            'Finish': format_dt(best['fin']),
            'Tipplers Used': used_str,
            'Total Duration': format_hhmm((best['fin'] - arrival) + timedelta(minutes=best['form'])),
            'Demurrage': format_hhmm(best['dem']),
        }
        row_res.update(flat_timings)
        results.append(row_res)

    return pd.DataFrame(results), tippler_state

def recalculate_reactive(edited_df, params, start_time):
    # Reactive calculation based on user manual edits in the DataFrame
    rows = []
    daily_stats = {}
    
    for _, row in edited_df.iterrows():
        # Parse times
        arr = pd.to_datetime(row['_Arrival_DT'])
        if arr.tzinfo is None: arr = IST.localize(arr)
        
        # Use 'Finish' key consistently
        # Handle cases where column might not exist safely, though logic guarantees 'Finish'
        fin_str = row.get('Finish', '') 
        fin_dt = restore_dt(fin_str, arr)
        
        # Form Mins
        form = float(row['_Form_Mins'])
        
        if pd.notnull(fin_dt):
            dur = (fin_dt - arr) + timedelta(minutes=form)
            dem = max(timedelta(0), dur - timedelta(hours=params['ft']))
            
            row['Total Duration'] = format_hhmm(dur)
            row['Demurrage'] = format_hhmm(dem)
            
            d_str = arr.strftime('%Y-%m-%d')
            daily_stats[d_str] = daily_stats.get(d_str, 0) + dem.total_seconds()
            
        rows.append(row)
        
    return pd.DataFrame(rows), daily_stats

# ==========================================
# 4. SIDEBAR INPUTS
# ==========================================
st.sidebar.header("⚙️ Configuration")
sim_params = {}

# Rates (Capacity Logic)
st.sidebar.subheader("🏗️ Capacities (Rates)")
c1, c2 = st.sidebar.columns(2)
sim_params['rt1'] = c1.number_input("Rate T1 (W/Hr)", 6.0, help="~11 wagons/batch")
sim_params['rt2'] = c2.number_input("Rate T2 (W/Hr)", 6.0, help="~11 wagons/batch")
sim_params['rt3'] = c1.number_input("Rate T3 (W/Hr)", 9.0, help="~30 wagons/batch")
sim_params['rt4'] = c2.number_input("Rate T4 (W/Hr)", 9.0, help="~30 wagons/batch")

# Shunting
st.sidebar.subheader("⏱️ Shunting & Delays")
sim_params['sa'] = st.sidebar.number_input("Line 8-10 Shunt (Min)", 25)
sim_params['sb'] = st.sidebar.number_input("Line 11 Shunt (Min)", 50)
st.sidebar.markdown("**Transfer Penalties:**")
sim_params['shunt_intra'] = st.sidebar.number_input("Partner Shunt (T1↔T2)", 15, help="Time to move balance wagons between pair")
sim_params['shunt_cross'] = st.sidebar.number_input("Cross Shunt (T1↔T3)", 45, help="Time to move balance wagons to other pair")

# Other
st.sidebar.subheader("📉 General")
sim_params['fa'] = st.sidebar.number_input("Formation A (Min)", 20)
sim_params['fb'] = st.sidebar.number_input("Formation B (Min)", 50)
sim_params['ft'] = st.sidebar.number_input("Free Time (Hrs)", 7.0)

# Downtime
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Downtime Manager")
if 'downtimes' not in st.session_state: st.session_state.downtimes = []
with st.sidebar.form("dt"):
    dt_t = st.selectbox("Tippler", ["T1", "T2", "T3", "T4"])
    dt_d = st.date_input("Date", datetime.now())
    dt_time = st.time_input("Time", datetime.now().time())
    dt_dur = st.number_input("Duration (Mins)", 60, step=30)
    if st.form_submit_button("Add Failure"):
        start = IST.localize(datetime.combine(dt_d, dt_time))
        st.session_state.downtimes.append({'Tippler': dt_t, 'Start': start, 'End': start+timedelta(minutes=dt_dur)})
        st.rerun()

if st.session_state.downtimes:
    st.sidebar.dataframe(pd.DataFrame(st.session_state.downtimes)[['Tippler', 'Start', 'End']])
    if st.sidebar.button("Clear Downtimes"):
        st.session_state.downtimes = []; st.rerun()

sim_params['downtimes'] = st.session_state.downtimes

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    # Cache Logic
    if 'last_id' not in st.session_state or st.session_state.last_id != uploaded_file.file_id:
        st.session_state.last_id = uploaded_file.file_id
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.upper()
        # Filter BOXN
        if 'LOAD TYPE' in df.columns:
            df = df[df['LOAD TYPE'].str.contains('BOXN', na=False)]
        df['wagon_count'] = df['TOTL UNTS'].apply(parse_wagons)
        df['arrival_dt'] = pd.to_datetime(df['EXPD ARVLTIME']).apply(to_ist)
        df = df.dropna(subset=['arrival_dt'])
        st.session_state.raw_df = df
        
        # Calc Start Time
        sim_params['start_time'] = df['arrival_dt'].min().replace(hour=0, minute=0, second=0)
        
        # Run Sim
        res_df, _ = run_simulation(df, sim_params)
        st.session_state.res_df = res_df

    if 'res_df' in st.session_state:
        st.markdown("### 📋 Optimized Schedule")
        
        # Config for Editor
        col_cfg = {
            "_Arrival_DT": None, "_Ready_DT": None, "_Form_Mins": None,
            "Strategy": st.column_config.TextColumn("Route Choice", width="medium", disabled=True),
            "Finish": st.column_config.TextColumn("Finish (dd-HH:MM)", help="Edit to update demurrage"),
            "Demurrage": st.column_config.TextColumn("Demurrage", disabled=True),
        }
        
        edited_df = st.data_editor(
            st.session_state.res_df,
            column_config=col_cfg,
            use_container_width=True,
            num_rows="fixed",
            key="main_editor"
        )
        
        # Reactive Updates
        final_df, daily = recalculate_reactive(edited_df, sim_params, None)
        
        # Forecast
        st.subheader("📊 Demurrage Forecast")
        stats_data = [{"Date": k, "Demurrage Cost": format_hhmm(timedelta(seconds=v))} for k,v in daily.items()]
        st.dataframe(pd.DataFrame(stats_data), use_container_width=False)
        
        # Download
        csv = final_df.drop(columns=[c for c in final_df.columns if c.startswith("_")]).to_csv(index=False).encode('utf-8')
        st.download_button("Download Schedule", csv, "optimized_schedule.csv", "text/csv")
