"""Mission sizing with empirical engine fits, exact ISA and shared flight physics."""
import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
import numpy as np
from aero import drag_terms, fin_area
from engine_choice import add_hardware_arguments, catalogue_engine, catalogue_candidates, nonnegative_weight
from mission_atmosphere import atmosphere_at_altitude, RHO_SL
from mission_path import flight_segments, point_drag, thrust_lapse, segment_loads, path_fuel
from missionsweep import MissionCase


@dataclass
class MissionModel:
    formulation: object
    variables: dict
    mission: MissionCase
    altitude: float
    segments: list


def build_mission(altitude, mission):
    """Build one candidate; engine enumeration happens in select_catalogue_engine."""
    from lcsolver import Formulation
    m = mission
    atm = atmosphere_at_altitude(altitude)
    if not np.isfinite(m.W_max) or m.W_max <= 0 or not np.isfinite(m.accel_margin) or m.accel_margin < 0:
        raise ValueError("weight cap must be positive; thrust margin must be finite and nonnegative")
    for name in ("W_prop_install", "W_pay", "W_wing_min", "W_fus_min"):
        if not np.isfinite(getattr(m, name)) or getattr(m, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not all(np.isfinite(v) and v >= 1 for v in (m.wing_weight_factor,m.fuselage_weight_factor)):
        raise ValueError("structural weight factors must be finite and at least 1")
    engine = catalogue_engine(m.engine_model) if m.engine_model else None
    f = Formulation()
    variables = {}
    def var(name, guess, units, description):
        variables[name] = f.Variable(name=name, guess=guess, units=units, description=description)
        return variables[name]
    W_TO = var("W_TO",65,"lbf","Takeoff gross weight")
    W_ZF = var("W_ZF",50,"lbf","Zero fuel weight")
    W_empty = var("W_empty",50,"lbf","Empty weight")
    W_wing = var("W_wing",4,"lbf","Wing structural estimate")
    W_eng = var("W_eng",engine.weight if engine else 20,"lbf","Catalogue engine weight")
    W_fus = var("W_fus",6,"lbf","Fuselage structural estimate")
    W_surf = var("W_surf",1,"lbf","Fin and strake structure")
    W_other = var("W_other",15,"lbf","Fixed systems plus other-weight allowance")
    W_f_climb = var("W_f_climb",10,"lbf","Start/takeoff plus climb/acceleration fuel")
    W_f_cruis = var("W_f_cruis",.15,"lbf","Dash fuel")
    W_f_res = var("W_f_res",3,"lbf","Reserve fuel")
    W_cr_start = var("W_cr_strt",55,"lbf","Weight at dash start")
    W_cr_end = var("W_cr_end",54,"lbf","Weight at dash end")
    S = var("S",4,"ft^2","Wing area")
    tau = var("tau",.04,"-","Thickness/chord")
    C_L = var("C_L",.06,"-","Dash lift coefficient")
    C_D = var("C_D",.07,"-","Dash drag coefficient")
    D = var("D",65,"lbf","Dash drag")
    T_seg = var("T_seg",65,"lbf","Dash thrust required")
    T_SL = var("T_SL",engine.thrust if engine else 200,"lbf","Sea-level static thrust")
    c_T = var("c_T",engine.tsfc if engine else 1.3,"1/hr","Static full-thrust TSFC approximation")
    c_sw = var("c_sw",.8,"-","Cosine of leading-edge sweep")
    L = var("L_fus",10,"ft","Fuselage length")
    d = var("d_fus",engine.diameter if engine else .7,"ft","Fuselage diameter")
    # aero.py takes normalized imperial numbers, so symbolic Reynolds numbers
    # and drag coefficients remain dimensionless.
    lb = f.Constant(name="lbf_ref",value=1,units="lbf")
    ft = f.Constant(name="ft_ref",value=1,units="ft")
    ft2 = f.Constant(name="ft2_ref",value=1,units="ft^2")
    sec = f.Constant(name="s_ref",value=1,units="s")
    per_hour = f.Constant(name="per_hr_ref",value=1,units="1/hr")
    w, sn, ln, dn, ct = W_TO/lb, S/ft2, L/ft, d/ft, c_T/per_hour
    speed = m.M * atm.speed_of_sound_ft_s
    q = .5 * atm.density_slug_ft3 * speed**2
    lapse = thrust_lapse(m,altitude,m.M)
    terms, extra = drag_terms(m,S=sn,tau=tau,c_sw=c_sw,L=ln,d=dn,
                             rho=atm.density_slug_ft3,V=speed,M=m.M,C_L=C_L)
    constraints = [
        W_TO/lb >= W_ZF/lb + W_f_climb/lb + W_f_cruis/lb + W_f_res/lb,
        W_ZF/lb >= m.W_pay + W_empty/lb,
        W_empty/lb >= W_wing/lb + W_eng/lb + W_fus/lb + W_surf/lb + W_other/lb + m.W_prop_install,
        W_wing/lb >= m.wing_weight_factor*m.K_wing*(m.N_ult*w)**.5*sn**.622*m.AR**.785*tau**-.4/c_sw,
        W_fus/lb >= m.fuselage_weight_factor*m.w_fus*np.pi*dn*ln,
        W_surf/lb >= m.w_surf*(extra["S_v"]+extra["S_str"]),
        W_other/lb >= m.W_fixed+m.f_other*w,
        W_f_res/lb >= m.f_res*W_ZF/lb,
        W_TO/lb >= W_cr_start/lb+W_f_climb/lb,
        W_cr_start/lb >= W_cr_end/lb+W_f_cruis/lb,
        W_cr_end/lb >= W_ZF/lb+W_f_res/lb,
        W_cr_start/lb == q*C_L*sn,
        C_D >= sum(terms.values()),
        D/lb >= q*C_D*sn,
        T_seg >= D,
        W_f_cruis/lb >= ct/3600*T_seg/lb*m.t_super,
        lapse*T_SL >= T_seg,
        w <= m.W_max,
        w <= .5*RHO_SL*m.V_stall**2*m.C_Lmax*sn,
        T_SL >= m.TW_TO*W_TO,
        c_sw <= 1,
        c_sw >= m.AR/np.sqrt(16+m.AR**2),
        tau >= m.tau_min,
        ln >= m.L_min,
        ln <= m.L_max,
        dn >= m.d_min,
    ]
    if m.W_wing_min: constraints.append(W_wing/lb >= m.W_wing_min)
    if m.W_fus_min: constraints.append(W_fus/lb >= m.W_fus_min)
    if engine:
        constraints += [W_eng/lb == engine.weight,T_SL/lb == engine.thrust,
                        ct == engine.tsfc,dn >= engine.diameter]
    else:
        constraints += [W_eng/lb >= (T_SL/lb-m.T_int)/m.T_slope,
                        W_eng/lb >= m.W_eng_min,W_eng/lb <= m.W_eng_max,
                        ct == m.tsfc_int+m.tsfc_slope*W_eng/lb,
                        dn >= m.d_coef*(W_eng/lb)**m.d_exp]
    segments = flight_segments(m,altitude)
    fuel_terms = []
    for i, segment in enumerate(segments):
        drag = var(f"D_path_{i}",30,"lbf",segment.phase+" maximum sampled drag")
        duration = var(f"dt_path_{i}",20,"s",segment.phase+" time")
        for point in segment.points:
            constraints.append(drag/lb >= point_drag(m,point,weight=w,S=sn,tau=tau,c_sw=c_sw,L=ln,d=dn))
        low, high = min(p.lapse for p in segment.points),max(p.lapse for p in segment.points)
        thrust = low*T_SL/lb
        # m*dV/dt=T-D; fixed-speed climb: W*dh/dt=(T-D)*V.
        # Use takeoff weight, maximum sampled drag, minimum thrust for timing.
        constraints += [drag/lb/thrust+w*segment.time_coefficient/(thrust*duration/sec) <= 1,
                        (1+m.accel_margin)*drag <= low*T_SL]
        if segment.minimum_time:
            constraints.append(duration/sec >= segment.minimum_time)
        fuel_terms.append(ct/3600*high*T_SL/lb*duration/sec)
    constraints.append(W_f_climb/lb >= m.f_TO*w+sum(fuel_terms))
    f.ConstraintList(constraints)
    f.Objective(W_TO)
    return MissionModel(f,variables,m,altitude,segments)


def solution_state(model,sol):
    m = model.mission
    r = {k:float(v.magnitude if hasattr(v,"magnitude") else v) for k,v in sol.variables().items()}
    atm = atmosphere_at_altitude(model.altitude)
    r.update(h=model.altitude,rho=atm.density_slug_ft3,a=atm.speed_of_sound_ft_s,
             lapse=thrust_lapse(m,model.altitude,m.M),W_prop_install=m.W_prop_install)
    r["c_sw"] = float(np.clip(r["c_sw"],0,1))
    r["S_v"] = fin_area(m,r["L_fus"],r["d_fus"])
    r["cos_sw"], r["sweep_deg"] = r["c_sw"],float(np.degrees(np.arccos(r["c_sw"])))
    r["Swet_fus"] = np.pi*r["d_fus"]*r["L_fus"]
    loads = segment_loads(m,model.segments,weight=r["W_TO"],S=r["S"],tau=r["tau"],
                          c_sw=r["c_sw"],L=r["L_fus"],d=r["d_fus"])
    r["path_fuel_check"],r["path_time_check"] = path_fuel(m,loads,r["W_TO"],r["T_SL"],r["c_T"])
    r["path_thrust_margin"] = min(r["T_SL"]*lo/drag-1 for _,drag,lo,_ in loads)
    r["path_time"] = sum(r[f"dt_path_{i}"] for i in range(len(loads)))
    if not np.isfinite(r["path_fuel_check"]) or abs(r["W_f_climb"] - m.f_TO*r["W_TO"] - r["path_fuel_check"]) > 1e-4:
        raise ValueError("post-solve path fuel does not match the numerical mission")
    if not 0 < r["W_TO"] <= m.W_max*(1+1e-6) or r["path_thrust_margin"] < m.accel_margin-1e-6:
        raise ValueError("post-solve weight or thrust-margin constraint failed")
    r["constraint_residual"] = constraint_residual(model)
    if r["constraint_residual"] > 1e-5:
        raise ValueError(f"original model constraint residual is too large: {r['constraint_residual']:.3g}")
    return r


def constraint_residual(model):
    """Check every original equation after solver writeback, not just fuel."""
    import pyomo.environ as pyo
    maximum = 0.0
    for row in model.formulation.component_data_objects(pyo.Constraint,active=True):
        body = float(pyo.value(row.body))
        if not np.isfinite(body):
            return float("inf")
        for bound, sign in ((row.lower,-1),(row.upper,1)):
            if bound is not None:
                limit = float(pyo.value(bound))
                maximum = max(maximum,max(0.0,sign*(body-limit))/(1+abs(limit)))
    return maximum


def empirical_start(model):
    """A second, small-aircraft start; this is a guess, never a constraint."""
    m = model.mission
    weight = min(m.W_max,max(20.0,2*m.W_eng_min))
    engine_weight = m.W_eng_min
    return dict(W_TO=weight,W_ZF=.85*weight,W_empty=.85*weight,
                W_eng=engine_weight,T_SL=m.T_slope*engine_weight+m.T_int,
                c_T=m.tsfc_int+m.tsfc_slope*engine_weight,
                S=2*weight/(RHO_SL*m.V_stall**2*m.C_Lmax),tau=max(.03,m.tau_min),
                c_sw=1.0,L_fus=m.L_min,d_fus=max(m.d_min,m.d_coef*engine_weight**m.d_exp),
                W_cr_strt=.9*weight,W_cr_end=.89*weight)


def solve_mission(model, initial_state=None):
    """Keep the lightest checked local solution from independent starts.

    The signomial empirical model has no global-optimality certificate.
    Catalogue candidates remain convex fixed-engine subproblems.
    """
    import lcsolver
    if model.mission.engine_model:
        sol = lcsolver.solve(model.formulation)
        return model, sol, solution_state(model,sol)
    starts = [("small-aircraft guess",empirical_start(model)),
              ("previous altitude" if initial_state else "nominal guess",initial_state)]
    best, attempts = None, []
    for label, seed in starts:
        candidate = build_mission(model.altitude,model.mission)
        if seed:
            for name, value in seed.items():
                if name in candidate.variables and np.isfinite(value) and value > 0:
                    candidate.variables[name].set_value(float(value))
        try:
            sol = lcsolver.solve(candidate.formulation,sensitivities=False)
            state = solution_state(candidate,sol)
            attempts.append((label,state["W_TO"],None))
            if best is None or state["W_TO"] < best[2]["W_TO"]:
                best = candidate,sol,state
        except (RuntimeError,ValueError) as error:
            attempts.append((label,None,str(error)))
    if best is None:
        raise RuntimeError("No checked local solution from either start: " + "; ".join(f"{name}: {error}" for name,_,error in attempts))
    best[1]["mission_starts"] = attempts
    return best


@dataclass
class EngineTrial:
    name: str
    weight: float | None
    message: str


@dataclass
class EngineSelection:
    model: MissionModel | None
    solution: object
    state: dict | None
    trials: list
    skipped: list


def solve_engine_candidate(altitude, mission, engine):
    import lcsolver
    model = build_mission(altitude, replace(mission, engine_model=engine.name))
    sol = lcsolver.solve(model.formulation, sensitivities=False)
    return model, sol, solution_state(model, sol)


def select_catalogue_engine(altitude, mission, candidates=None, progress=None):
    """Optimize aircraft geometry separately for each complete real engine.

    Compare total takeoff weight, not engine weight. A failed solve is recorded
    as such and never establishes either feasibility or physical infeasibility.
    """
    atmosphere_at_altitude(altitude)
    skipped = []
    if candidates is None:
        candidates, skipped = catalogue_candidates()
    result = EngineSelection(None, None, None, [], skipped)
    for engine in candidates:
        try:
            model, sol, state = solve_engine_candidate(altitude, mission, engine)
            trial = EngineTrial(engine.name, state["W_TO"], "feasible")
            if result.state is None or state["W_TO"] < result.state["W_TO"]:
                result.model, result.solution, result.state = model, sol, state
        except (RuntimeError, ValueError) as error:
            trial = EngineTrial(engine.name, None, str(error))
        result.trials.append(trial)
        if progress is not None:
            progress(trial)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("altitude",nargs="?",type=float,help="geometric ft; omitted: search 0-50 kft")
    parser.add_argument("k_ram",nargs="?",type=float,help="optional Mach-1 density-fit coefficient")
    parser.add_argument("--altitude-step",type=float,default=2500)
    parser.add_argument("--max-weight",type=nonnegative_weight,default=80,help="design weight cap, lb")
    parser.add_argument("--continuous-engine-fit",action="store_true",help="use empirical engine fits (the default)")
    parser.add_argument("--accel-margin",type=float,default=.10,help="minimum excess thrust / drag across sampled path")
    parser.add_argument("--wing-weight-min",type=nonnegative_weight,default=0)
    parser.add_argument("--fuselage-weight-min",type=nonnegative_weight,default=0)
    parser.add_argument("--wing-weight-factor",type=float,default=MissionCase.wing_weight_factor,
                        help="multiplier on the wing correlation (at least 1; default 2)")
    parser.add_argument("--fuselage-weight-factor",type=float,default=MissionCase.fuselage_weight_factor,
                        help="multiplier on shell/frame weight (at least 1; default 1.5)")
    parser.add_argument("--initial-state",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--state-output",type=Path,help=argparse.SUPPRESS)
    add_hardware_arguments(parser)
    parser.set_defaults(engine=None)
    parser.epilog = "Default: empirical engine fits. --engine auto compares the catalogue; --engine NAME pins one engine."
    args = parser.parse_args()
    try:
        if args.continuous_engine_fit and "--engine" in sys.argv:
            parser.error("choose either --engine or --continuous-engine-fit")
        if args.max_weight <= 0 or not np.isfinite(args.accel_margin) or args.accel_margin < 0:
            parser.error("weight cap must be positive; thrust margin finite and nonnegative")
        name = None if args.continuous_engine_fit else args.engine
        if name and name != "auto": catalogue_engine(name)
        if args.altitude is None:
            from altitude_search import run_altitude_search
            return run_altitude_search(Path(__file__).resolve(),sys.argv[1:],args.altitude_step)
        m = MissionCase(engine_model=name,W_max=args.max_weight,L_min=6,W_pay=args.payload_weight,
                        W_prop_install=args.installation_weight,lapse_override=args.lapse,
                        accel_margin=args.accel_margin,W_wing_min=args.wing_weight_min,
                        W_fus_min=args.fuselage_weight_min,wing_weight_factor=args.wing_weight_factor,
                        fuselage_weight_factor=args.fuselage_weight_factor)
        if args.k_ram is not None: m.K_ram = args.k_ram
        atmosphere_at_altitude(args.altitude)
        if name != "auto":
            model = build_mission(args.altitude,m)
    except ValueError as error:
        parser.error(str(error))
    import lcsolver
    print(f"Atmosphere: exact ISA at {args.altitude:g} geometric ft")
    print(f"Weight cap: {m.W_max:g} lb; path excess-thrust margin: {m.accel_margin:.0%}")
    print(f"Structural allowances: wing {m.wing_weight_factor:g}x; fuselage {m.fuselage_weight_factor:g}x (provisional)")
    print("Lapse: "+("constant user scenario across the full path" if args.lapse is not None
                        else "screening-cycle shape with Mach-1 density fit; not engine validation"))
    if name == "auto":
        print("Comparing catalogue engines by optimized aircraft takeoff weight", flush=True)
        print("Candidates share the screening lapse assumption; engine-specific off-design maps are unavailable.", flush=True)
        def report_trial(trial):
            outcome = f"{trial.weight:.4f} lb MTOW" if trial.weight is not None else "no verified feasible solution"
            print(f"  {trial.name}: {outcome}", flush=True)
        selection = select_catalogue_engine(args.altitude,m,progress=report_trial)
        for skipped_name, reason in selection.skipped:
            print(f"  Skipped {skipped_name}: {reason}")
        if selection.model is None:
            print("No catalogue candidate produced a verified feasible solution.",file=sys.stderr)
            for trial in selection.trials:
                print(f"  {trial.name}: {trial.message}",file=sys.stderr)
            return 1
        model, sol, r = selection.model, selection.solution, selection.state
        m = model.mission
        if any(trial.weight is None for trial in selection.trials):
            print("Selecting among checked, converged designs; failed solves are not proofs of infeasibility.")
        print(f"Selected engine: {m.engine_model}")
        engine = catalogue_engine(m.engine_model)
        print(f"  {engine.weight:g} lb, {engine.thrust:g} lbf SLS, TSFC {engine.tsfc:g} lb/(lbf hr), diameter {engine.diameter*12:.3f} in")
    else:
        print(f"Engine: {name or 'empirical fits (continuous conceptual engine)'}")
        try:
            initial = None
            if args.initial_state is not None:
                saved = json.loads(args.initial_state.read_text())
                initial = saved.get("variables",saved)
            model, sol, r = solve_mission(model,initial)
        except (RuntimeError, ValueError) as error:
            print(f"No converged feasible design: {error}",file=sys.stderr)
            return 1
    if args.state_output is not None:
        args.state_output.write_text(json.dumps({"variables":r,"mode":"empirical" if name is None else "catalogue"}))
    summary = sol.summary()
    if name is None:
        print("Empirical sizing: best checked local solution; global optimality and hardware feasibility are not certified.")
        for label, weight, error in sol["mission_starts"]:
            print(f"  Start {label}: {weight:.4f} lb" if weight is not None else f"  Start {label}: no checked solution")
        if r["W_eng"] <= m.W_eng_min*(1+1e-6):
            print(f"Engine fit is at its {m.W_eng_min:g} lb lower bound; structural and installation weights remain assumptions.")
        summary = summary.replace("Status: optimal","Status: local solution (not a global certificate)")
    print(summary)
    print(f"Path time: {r['path_time']:.1f} s; minimum sampled thrust margin: {r['path_thrust_margin']:.1%}")
    print(f"Path fuel numerical check: {r['path_fuel_check']:.4f} lb")
    print("Structural weights remain empirical; stiffness, flutter and thermal limits are unvalidated.")
    from breakdown import print_breakdown
    print_breakdown(r,asdict(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
