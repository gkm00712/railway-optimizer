import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import pytz

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Railway Logic Optimizer (Final)", layout="wide")
st.title("🚂 BOXN Demurrage Optimizer: Capacity & Rescue Logic")
st.markdown("""
**Logic Implemented:**
1.  **Capacity Constraints:** T1/T2 take max **11 wagons/batch**; T3/T4 take max **30 wagons/batch**.
2.  **Parallel Unloading:** Next batch is ready **10 mins** after previous batch starts (allowing T1 & T2 to work simultaneously).
3.  **Breakdown Rescue:** If a tippler fails, the system auto-calculates if moving to the Partner (Intra) or Cross (Inter) is faster than waiting.
""")

# Define IST Timezone
IST = pytz.timezone('Asia/Kolkata')

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def to_ist(dt):
    """Ensures a datetime object is in IST."""
    if pd.isnull(dt): return pd.NaT
    if dt.tzinfo is None:
        return IST.localize(dt)
    return dt.astimezone(IST)

def parse_wagons(val):
    try:
        if '+' in str(val):
            parts = str(val).split('+')
            return sum(int(p) for p in parts if p.strip().isdigit())
        return int(float(val))
    except: return 0

def format_dt(dt):
    if pd.isnull(dt): return ""
    if dt.tzinfo is None: dt = IST.localize(dt)
    return dt.strftime('%d-%H:%M')

def format_duration_hhmm(delta):
    if pd.isnull(delta): return ""
    total_seconds = int(delta.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{sign}{hours:02d}:{minutes:02d}"

# ==========================================
# 3. ADVANCED SIMULATION LOGIC
# ==========================================

def get_next_available_start(tippler_id, proposed_start, downtime_list):
    """
    Checks if proposed_start is inside a downtime. 
    If yes, returns the end of that downtime. 
    """
    if proposed_start.tzinfo is None:
        proposed_start = IST.localize(proposed_start)
        
    relevant_dts = [d for d in downtime_list if d['Tippler'] == tippler_id]
    relevant_dts.sort(key=lambda x: x['Start'])
    
    current_check = proposed_start
    changed = True
    while changed:
        changed = False
        for dt in relevant_dts:
            if dt['Start'] <= current_check < dt['End']:
                current_check = dt['End']
                changed = True
    return current_check

def find_interruption(tippler_id, start_time, end_time, downtime_list):
    """
    Checks if a downtime STARTS strictly between start_time and end_time.
    """
    relevant_dts = [d for d in downtime_list if d['Tippler'] == tippler_id]
    earliest_int = None
    for dt in relevant_dts:
        if start_time < dt['Start'] < end_time:
            if earliest_int is None or dt['Start'] < earliest_int:
                earliest_int = dt['Start']
    return earliest_int

def solve_tippler_allocation_advanced(total_wagons, allowed_tipplers, ready_time, 
                                     tippler_state, downtime_list, rates, 
                                     shunt_intra, shunt_inter, split_delay):
    """
    Full Simulation Engine handling:
    1. 11 vs 30 Wagon Batches
    2. Parallel starts (Split Delay)
    3. Breakdown Rescue (Wait vs Intra vs Inter)
    """
    
    # --- CAPACITY DEFINITIONS ---
    batch_caps = {'T1': 11, 'T2': 11, 'T3': 30, 'T4': 30}
    
    # Maps for rescue logic
    pair_map = {'T1': 'T2', 'T2': 'T1', 'T3': 'T4', 'T4': 'T3'}
    group_map = {'T1': ['T3','T4'], 'T2': ['T3','T4'], 'T3': ['T1','T2'], 'T4': ['T1','T2']}
    
    wagons_remaining = total_wagons
    
    # 'current_ready_time' is when the NEXT batch of wagons is physically ready to be shunted/placed
    current_ready_time = ready_time 
    
    logs = []
    
    # Work on a local copy of state so we don't mess up the global state during trial runs
    local_state = tippler_state.copy()
    detailed_timings = {}
    total_idle = timedelta(0)
    
    iteration_guard = 0
    
    # --- BATCH PROCESSING LOOP ---
    while wagons_remaining > 0 and iteration_guard < 50:
        iteration_guard += 1
        
        # A. SELECTION PHASE: Which tippler allows the earliest START for the next batch?
        best_t = None
        earliest_possible_start = None
        
        # Sort allowed tipplers by who is free soonest
        candidates = sorted(allowed_tipplers, key=lambda t: local_state[t])
        
        # We pick the candidate that gives the earliest *actual* start time (considering downtimes)
        for t in candidates:
            mach_free = local_state[t]
            if mach_free.tzinfo is None: mach_free = IST.localize(mach_free)
            
            # Start = Max(Wagons Ready, Machine Free) -> Adjusted for Downtime
            actual_start = get_next_available_start(t, max(current_ready_time, mach_free), downtime_list)
            
            if earliest_possible_start is None or actual_start < earliest_possible_start:
                earliest_possible_start = actual_start
                best_t = t
        
        # B. EXECUTION PHASE
        t_active = best_t
        start_time = earliest_possible_start
        
        # Determine Batch Size (11 for T1/T2, 30 for T3/T4)
        cap = batch_caps[t_active]
        w_load = min(wagons_remaining, cap)
        rate = rates[t_active]
        
        # Calculate Idle for this specific machine assignment
        prev_end = local_state[t_active]
        if prev_end.tzinfo is None: prev_end = IST.localize(prev_end)
        idle_this_step = max(timedelta(0), start_time - prev_end)
        
        # Theoretical Finish
        duration = timedelta(hours=(w_load / rate))
        end_time_theory = start_time + duration
        
        # C. BREAKDOWN CHECK
        interruption_pt = find_interruption(t_active, start_time, end_time_theory, downtime_list)
        
        if interruption_pt:
            # === RESCUE LOGIC TRIGGERED ===
            
            # 1. Calculate partial work
            time_worked = interruption_pt - start_time
            wagons_done = int((time_worked.total_seconds() / 3600) * rate)
            wagons_stuck = w_load - wagons_done
            
            logs.append(f"{t_active}: Broken at {format_dt(interruption_pt)}. Done: {wagons_done}, Stuck: {wagons_stuck}")
            
            # Update state for partial work
            detailed_timings[f"{t_active}_Start"] = detailed_timings.get(f"{t_active}_Start", start_time)
            # Machine blocked until interruption
            local_state[t_active] = interruption_pt 
            
            if wagons_stuck > 0:
                # OPTION 1: WAIT (Resume on same machine after fix)
                resume_time = get_next_available_start(t_active, interruption_pt, downtime_list)
                dur_rem = timedelta(hours=(wagons_stuck / rates[t_active]))
                fin_wait = resume_time + dur_rem
                
                # OPTION 2: INTRA-SHUNT (Move to partner)
                t_partner = pair_map[t_active]
                # Partner ready time = Break Time + Intra Shunt
                ready_intra = interruption_pt + timedelta(minutes=shunt_intra)
                start_intra = get_next_available_start(t_partner, max(ready_intra, local_state[t_partner]), downtime_list)
                fin_intra = start_intra + timedelta(hours=(wagons_stuck / rates[t_partner]))
                
                # OPTION 3: INTER-SHUNT (Move to other group)
                ready_inter = interruption_pt + timedelta(minutes=shunt_inter)
                best_inter_fin = pd.Timestamp.max.replace(tzinfo=IST)
                best_inter_t = None
                
                for t_other in group_map[t_active]:
                    start_o = get_next_available_start(t_other, max(ready_inter, local_state[t_other]), downtime_list)
                    fin_o = start_o + timedelta(hours=(wagons_stuck / rates[t_other]))
                    if fin_o < best_inter_fin:
                        best_inter_fin = fin_o
                        best_inter_t = t_other
                
                # DECISION
                choices = {'Wait': fin_wait, 'Intra': fin_intra, 'Inter': best_inter_fin}
                best_strat = min(choices, key=choices.get)
                logs.append(f"-> Rescue: {best_strat} (Fin: {format_dt(choices[best_strat])})")
                
                batch_finish_time = choices[best_strat]
                
                # Apply Decision to State
                if best_strat == 'Wait':
                    local_state[t_active] = fin_wait
                    detailed_timings[f"{t_active}_End"] = fin_wait
                elif best_strat == 'Intra':
                    local_state[t_partner] = fin_intra
                    detailed_timings[f"{t_partner}_Start"] = detailed_timings.get(f"{t_partner}_Start", start_intra)
                    detailed_timings[f"{t_partner}_End"] = fin_intra
                    # Active tippler stays free at interruption point
                else:
                    local_state[best_inter_t] = best_inter_fin
                    detailed_timings[f"{best_inter_t}_Start"] = detailed_timings.get(f"{best_inter_t}_Start", best_inter_fin - timedelta(hours=wagons_stuck/rates[best_inter_t]))
                    detailed_timings[f"{best_inter_t}_End"] = best_inter_fin

            else:
                batch_finish_time = interruption_pt
                local_state[t_active] = interruption_pt
                detailed_timings[f"{t_active}_End"] = interruption_pt

        else:
            # === STANDARD SUCCESS ===
            local_state[t_active] = end_time_theory
            detailed_timings[f"{t_active}_Idle"] = detailed_timings.get(f"{t_active}_Idle", timedelta(0)) + idle_this_step
            detailed_timings[f"{t_active}_Start"] = detailed_timings.get(f"{t_active}_Start", start_time)
            detailed_timings[f"{t_active}_End"] = end_time_theory
            total_idle += idle_this_step
            batch_finish_time = end_time_theory

        # D. PREPARE FOR NEXT LOOP
        wagons_remaining -= w_load
        
        # PARALLELING LOGIC:
        # As soon as this batch *starts*, the next split is being prepared.
        # It arrives at the tippler (or partner) after 'split_delay' minutes relative to START.
        current_ready_time = start_time + timedelta(minutes=split_delay)

    # Compile Final Stats
    final_finish = ready_time
    used_list = []
    for t in ['T1', 'T2', 'T3', 'T4']:
        if f"{t}_End" in detailed_timings:
            final_finish = max(final_finish, detailed_timings[f"{t}_End"])
            used_list.append(t)
            
    return final_finish, ", ".join(sorted(used_list)), detailed_timings, total_idle, "; ".join(logs)


def run_simulation(df, params):
    # Unpack params
    rates = {'T1': params['rt1'], 'T2': params['rt2'], 'T3': params['rt3'], 'T4': params['rt4']}
    s_a, s_b = params['sa'], params['sb']
    shunt_intra = params['shunt_intra'] 
    shunt_inter = params['shunt_inter']
    f_a, f_b = params['fa'], params['fb']
    ft_hours = params['ft']
    split_delay = params['split_delay']
    downtimes = params['downtimes']

    df = df.copy()
    if 'LOAD TYPE' in df.columns:
        df = df[df['LOAD TYPE'].astype(str).str.strip().str.upper().str.startswith('BOXN')]
    
    df['wagon_count'] = df['TOTL UNTS'].apply(parse_wagons)
    df['exp_arrival_dt'] = pd.to_datetime(df['EXPD ARVLTIME'], errors='coerce').apply(to_ist)
    if 'STTS CODE' in df.columns and 'STTS TIME' in df.columns:
        df['stts_time_dt'] = pd.to_datetime(df['STTS TIME'], errors='coerce').apply(to_ist)
        df['arrival_dt'] = df.apply(lambda r: r['stts_time_dt'] if str(r.get('STTS CODE')).strip()=='PL' and pd.notnull(r['stts_time_dt']) else r['exp_arrival_dt'], axis=1)
    else:
        df['arrival_dt'] = df['exp_arrival_dt']

    df = df.dropna(subset=['arrival_dt']).sort_values('arrival_dt').reset_index(drop=True)
    if df.empty: return pd.DataFrame(), None

    # Init State
    first_arrival = df['arrival_dt'].min()
    sim_start = first_arrival.replace(hour=0, minute=0, second=0, microsecond=0)
    if sim_start.tzinfo is None: sim_start = IST.localize(sim_start)

    tippler_state = {k: sim_start for k in ['T1', 'T2', 'T3', 'T4']}
    line_groups = {
        'Group_Lines_8_10': {'capacity': 2, 'clearance_mins': 50, 'line_free_times': []}, 
        'Group_Line_11': {'capacity': 1, 'clearance_mins': 100, 'line_free_times': []}
    }
    assignments = []

    def get_line_entry(grp, arr):
        g = line_groups[grp]
        active = sorted([t for t in g['line_free_times'] if t > arr])
        if len(active) < g['capacity']: return arr
        return active[len(active) - g['capacity']]

    for _, rake in df.iterrows():
        options = []
        
        # --- OPTION A: Natural (Lines 8-10 -> T1/T2) ---
        entry_A = get_line_entry('Group_Lines_8_10', rake['arrival_dt'])
        ready_A = entry_A + timedelta(minutes=s_a)
        
        # Logic: T1/T2
        fin_A, used_A, tim_A, idle_A, log_A = solve_tippler_allocation_advanced(
            rake['wagon_count'], ['T1', 'T2'], ready_A, tippler_state, downtimes, rates, 
            shunt_intra, shunt_inter, split_delay
        )
        dem_A = max(timedelta(0), (fin_A - rake['arrival_dt']) + timedelta(minutes=f_a) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Nat_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_A, 
            'fin': fin_A, 'used': used_A, 'timings': tim_A, 'dem': dem_A, 'type': 'Standard (A)', 'form': f_a
        })

        # --- OPTION B: Natural (Line 11 -> T3/T4) ---
        entry_B = get_line_entry('Group_Line_11', rake['arrival_dt'])
        ready_B = entry_B + timedelta(minutes=s_b)
        
        # Logic: T3/T4
        fin_B, used_B, tim_B, idle_B, log_B = solve_tippler_allocation_advanced(
            rake['wagon_count'], ['T3', 'T4'], ready_B, tippler_state, downtimes, rates, 
            shunt_intra, shunt_inter, split_delay
        )
        dem_B = max(timedelta(0), (fin_B - rake['arrival_dt']) + timedelta(minutes=f_b) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Nat_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_B, 
            'fin': fin_B, 'used': used_B, 'timings': tim_B, 'dem': dem_B, 'type': 'Standard (B)', 'form': f_b
        })

        # --- OPTION C: Cross A->B (Lines 8-10 -> T3/T4) ---
        ready_AX = ready_A + timedelta(minutes=shunt_inter)
        fin_AX, used_AX, tim_AX, idle_AX, log_AX = solve_tippler_allocation_advanced(
            rake['wagon_count'], ['T3', 'T4'], ready_AX, tippler_state, downtimes, rates, 
            shunt_intra, shunt_inter, split_delay
        )
        dem_AX = max(timedelta(0), (fin_AX - rake['arrival_dt']) + timedelta(minutes=f_a) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Cross_A', 'grp': 'Group_Lines_8_10', 'entry': entry_A, 'ready': ready_AX, 
            'fin': fin_AX, 'used': used_AX, 'timings': tim_AX, 'dem': dem_AX, 'type': 'Cross (A->B)', 'form': f_a
        })

        # --- OPTION D: Cross B->A (Line 11 -> T1/T2) ---
        ready_BX = ready_B + timedelta(minutes=shunt_inter)
        fin_BX, used_BX, tim_BX, idle_BX, log_BX = solve_tippler_allocation_advanced(
            rake['wagon_count'], ['T1', 'T2'], ready_BX, tippler_state, downtimes, rates, 
            shunt_intra, shunt_inter, split_delay
        )
        dem_BX = max(timedelta(0), (fin_BX - rake['arrival_dt']) + timedelta(minutes=f_b) - timedelta(hours=ft_hours))
        options.append({
            'id': 'Cross_B', 'grp': 'Group_Line_11', 'entry': entry_B, 'ready': ready_BX, 
            'fin': fin_BX, 'used': used_BX, 'timings': tim_BX, 'dem': dem_BX, 'type': 'Cross (B->A)', 'form': f_b
        })

        # --- PICK BEST ---
        best_opt = sorted(options, key=lambda x: (x['dem'], x['fin']))[0]

        # Update State
        for k, v in best_opt['timings'].items():
            if 'End' in k:
                t_id = k.split('_')[0]
                tippler_state[t_id] = max(tippler_state[t_id], v)
        
        clr = line_groups[best_opt['grp']]['clearance_mins']
        line_groups[best_opt['grp']]['line_free_times'].append(best_opt['entry'] + timedelta(minutes=clr))

        tot_dur = (best_opt['fin'] - rake['arrival_dt']) + timedelta(minutes=best_opt['form'])
        row = {
            'Rake': rake['RAKE NAME'], 'Wagons': rake['wagon_count'],
            'Arrival': format_dt(rake['arrival_dt']), 'Optimization Type': best_opt['type'],
            'Finish Unload': format_dt(best_opt['fin']), 'Total Duration': format_duration_hhmm(tot_dur),
            'Demurrage': format_duration_hhmm(best_opt['dem']), 'Tipplers': best_opt['used']
        }
        for t in ['T1','T2','T3','T4']:
            row[f"{t} Start"] = format_dt(best_opt['timings'].get(f"{t}_Start", pd.NaT))
            row[f"{t} End"] = format_dt(best_opt['timings'].get(f"{t}_End", pd.NaT))
        
        assignments.append(row)

    return pd.DataFrame(assignments), sim_start

# ==========================================
# 4. SIDEBAR PARAMS
# ==========================================
st.sidebar.header("⚙️ Settings")
sim_params = {}
sim_params['rt1'] = st.sidebar.number_input("T1 Rate", value=6.0)
sim_params['rt2'] = st.sidebar.number_input("T2 Rate", value=6.0)
sim_params['rt3'] = st.sidebar.number_input("T3 Rate", value=9.0)
sim_params['rt4'] = st.sidebar.number_input("T4 Rate", value=9.0)
st.sidebar.markdown("---")
sim_params['sa'] = st.sidebar.number_input("Base Shunt A", value=25.0)
sim_params['sb'] = st.sidebar.number_input("Base Shunt B", value=50.0)
sim_params['shunt_intra'] = st.sidebar.number_input("Intra-Shunt (T1<>T2)", value=15.0)
sim_params['shunt_inter'] = st.sidebar.number_input("Inter-Shunt (T1<>T3)", value=45.0)
sim_params['split_delay'] = st.sidebar.number_input("Split Delay (Mins)", value=10.0)
sim_params['fa'] = st.sidebar.number_input("Form A", value=20.0)
sim_params['fb'] = st.sidebar.number_input("Form B", value=50.0)
sim_params['ft'] = st.sidebar.number_input("Free Time (Hrs)", value=7.0)

# DOWNTIMES
st.sidebar.markdown("---")
if 'downtimes' not in st.session_state: st.session_state.downtimes = []
with st.sidebar.form("downtime_form"):
    dt_tippler = st.selectbox("Tippler", ["T1", "T2", "T3", "T4"])
    dt_date = st.date_input("Date", value=datetime.now(IST).date())
    dt_time = st.time_input("Time", value=datetime.now(IST).time())
    dt_dur = st.number_input("Duration (Mins)", value=60, step=15)
    if st.form_submit_button("Add Downtime"):
        start = IST.localize(datetime.combine(dt_date, dt_time))
        st.session_state.downtimes.append({"Tippler": dt_tippler, "Start": start, "End": start + timedelta(minutes=dt_dur)})
        st.rerun()

sim_params['downtimes'] = st.session_state.downtimes

# ==========================================
# 5. RUN
# ==========================================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    uploaded_file.seek(0)
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip().str.upper()
    
    res, start = run_simulation(df_raw, sim_params)
    
    if not res.empty:
        st.dataframe(res.style.apply(lambda x: ['background: #ffeeba' if 'Cross' in str(v) else '' for v in x], subset=['Optimization Type']))
        st.download_button("Download CSV", res.to_csv(index=False).encode('utf-8'), "schedule.csv", "text/csv")
    else:
        st.warning("No BOXN data found.")
