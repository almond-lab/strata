"""Scoped convergence forecasts and quadratic quantization models.

Forecast probabilities are fitted-model estimates, never calibrated certificates.
Exact grid results certify only the supplied rational quadratic and finite grids;
they are not lower bounds on an unmodelled nonlinear numerical pipeline.
"""
from __future__ import annotations

from fractions import Fraction
from decimal import Decimal
from itertools import product
from math import prod
import json
import re

import numpy as np

from .model import StrataError, digest, canonical


def _finite(value, name):
    if isinstance(value, bool):raise StrataError(name+' must be finite numeric data')
    try:result=float(value)
    except (TypeError,ValueError,OverflowError) as exc:raise StrataError(name+' must be finite numeric data') from exc
    if not np.isfinite(result):raise StrataError(name+' must be finite numeric data')
    return result


def _identity(checkpoint,objective,unit):
    if any(not isinstance(value,str) or not value.strip() for value in (checkpoint,objective,unit)):
        raise StrataError('checkpoint, objective and unit must be explicit nonempty identifiers')
    return dict(checkpoint=checkpoint,objective=objective,unit=unit)


def _match(row,identity):
    if any(row.get(key)!=value for key,value in identity.items()):
        raise StrataError('measurement checkpoint/objective/unit differs from frozen study')


def _fits(steps,values):
    """Fit small monotone trend families, choosing by temporal holdout error."""
    start=float(steps[0]);scale=max(float(steps[-1]-steps[0]),1.)
    t=(steps-start)/scale
    families=[('constant',0.) ,('linear',0.)]+[('exponential',float(k)) for k in (.1,.25,.5,1,2,4,8)]+[('power',float(k)) for k in (.25,.5,1,2,4)]
    def basis(family,k,x):
        if family=='constant':return np.zeros_like(x)
        if family=='linear':return x
        if family=='exponential':return np.exp(-k*x)
        return np.power(1+x,-k)
    def coefficients(family,k,x,y):
        if family=='constant':return np.array([float(np.mean(y)),0.])
        design=np.column_stack([np.ones(len(x)),basis(family,k,x)])
        beta=np.linalg.lstsq(design,y,rcond=None)[0]
        if family=='linear' and beta[1]>0 or family not in {'linear','constant'} and beta[1]<0:
            return np.array([float(np.mean(y)),0.])
        return beta
    held=max(1,min(2,len(steps)-3));cut=len(steps)-held;models=[]
    for family,k in families:
        train=coefficients(family,k,t[:cut],values[:cut])
        residual=values[cut:]-(train[0]+train[1]*basis(family,k,t[cut:]))
        score=float(np.mean(residual**2));beta=coefficients(family,k,t,values)
        full_residual=values-(beta[0]+beta[1]*basis(family,k,t))
        models.append(dict(family=family,parameter=k,intercept=float(beta[0]),coefficient=float(beta[1]),
            start=start,scale=scale,holdout_mse=score,residuals=full_residual.tolist()))
    models.sort(key=lambda m:(m['holdout_mse'],m['family'],m['parameter']))
    return models


def _predict(model,steps):
    x=(np.asarray(steps)-model['start'])/model['scale'];family=model['family'];k=model['parameter']
    basis=np.zeros_like(x) if family=='constant' else x if family=='linear' else np.exp(-k*x) if family=='exponential' else np.power(1+x,-k)
    return model['intercept']+model['coefficient']*basis


def forecast_convergence(runs, *, checkpoint,objective,unit,target,horizon,direction='min',
                         bootstrap_samples=1000,seed=0,band_level=.9,min_probability_runs=5):
    """Bootstrap independent measured trajectories under finite/asymptotic fits.

    A run is ``{run_id, independent: true, checkpoint, objective, unit,
    points: [{step,value}, ...]}``. Independence is a caller declaration, not a
    theorem established by distinct IDs. Bands distinguish the mean trend from
    a new same-population run. No stopping or resource decision is executed.
    """
    identity=_identity(checkpoint,objective,unit);target=_finite(target,'target');horizon=_finite(horizon,'horizon')
    if direction not in {'min','max'} or horizon<=0 or type(bootstrap_samples)is not int or bootstrap_samples<32 or type(seed)is not int or not 0<band_level<1 or type(min_probability_runs)is not int or min_probability_runs<3:
        raise StrataError('invalid convergence forecast budget or direction')
    normalized=[];ids=set();sign=1 if direction=='min' else -1
    for raw in runs:
        if not isinstance(raw,dict):raise StrataError('run must be a measured trajectory object')
        _match(raw,identity);rid=raw.get('run_id')
        if not isinstance(rid,str) or not rid or rid in ids:raise StrataError('independent run IDs must be distinct nonempty strings')
        ids.add(rid);points=raw.get('points')
        if not isinstance(points,list) or len(points)<5:raise StrataError('each trajectory needs at least five measured points')
        if any(not isinstance(p,dict) for p in points):raise StrataError('trajectory points must contain step and value')
        values=[dict(step=_finite(p.get('step'),'step'),value=_finite(p.get('value'),'value')) for p in points]
        if any(b['step']<=a['step'] for a,b in zip(values,values[1:])):raise StrataError('trajectory steps must increase strictly')
        normalized.append(dict(run_id=rid,independent=raw.get('independent')is True,points=values))
    if not normalized:raise StrataError('at least one measured trajectory is required')
    normalized.sort(key=lambda r:r['run_id']);n=len(normalized)
    source_hash=digest(dict(identity=identity,runs=normalized))
    model_hash=digest(dict(identity=identity,runs=normalized,target=target,horizon=horizon,direction=direction,
        bootstrap_samples=bootstrap_samples,seed=seed,band_level=band_level,min_probability_runs=min_probability_runs,
        families='constant/linear/exponential/power-v1'))
    result=dict(**identity,model_hash=model_hash,kind='fitted_trend',evidence_kind='fitted_trend',model_kind='convergence_forecast',source_hash=source_hash,
        direction=direction,target=target,horizon=horizon,run_count=n,seed=seed,bootstrap_samples=bootstrap_samples,
        probability_kind='fitted_model_estimate; not a calibrated certificate',calibrated_certificate=False,
        scope='declared independent repeats of one frozen checkpoint/objective/unit; no guarantee for a different pipeline or a particular continuation',
        independence='caller-declared; run identifiers alone do not establish independence',decision_authority='forecast only; observed predeclared rules remain separate')
    if n<3 or not all(r['independent'] for r in normalized):
        return dict(result,status='insufficient_evidence',target_probability=None,bands=None,
                    reason='at least three declared independent runs are needed for run-level uncertainty bands')
    fits=[]
    for run in normalized:
        steps=np.array([p['step'] for p in run['points']]);values=sign*np.array([p['value'] for p in run['points']])
        fits.append(_fits(steps,values))
    last=max(r['points'][-1]['step'] for r in normalized);future=np.linspace(last,last+horizon,21)
    rng=np.random.default_rng(seed);mean_samples=[];predictive_samples=[]
    model_weights=[]
    for models in fits:
        errors=np.array([m['holdout_mse'] for m in models]);scale=max(float(np.median(errors)),1e-20)
        weights=np.exp(-(errors-errors.min())/scale);model_weights.append(weights/weights.sum())
    for _ in range(bootstrap_samples):
        sampled=rng.integers(n,size=n);curves=[];chosen_models=[]
        for index in sampled:
            model=fits[index][int(rng.choice(len(fits[index]),p=model_weights[index]))]
            curves.append(_predict(model,future));chosen_models.append(model)
        curves=np.array(curves)
        if not np.isfinite(curves).all():raise StrataError('forecast extrapolation exceeds finite numerical range')
        mean_samples.append(sign*curves.mean(axis=0))
        selected=int(rng.integers(n));residuals=np.array(chosen_models[selected]['residuals']);residuals-=residuals.mean()
        predictive_samples.append(sign*(curves[selected]+float(rng.choice(residuals))))
    mean_samples=np.array(mean_samples);predictive_samples=np.array(predictive_samples);alpha=(1-band_level)/2
    def bands(samples):
        low,median,high=np.quantile(samples,[alpha,.5,1-alpha],axis=0)
        return dict(level=band_level,steps=future.tolist(),lower=low.tolist(),median=median.tolist(),upper=high.tolist())
    variation=len({digest(r['points']) for r in normalized})
    sufficient=n>=min_probability_runs and variation>=3
    success=predictive_samples[:,-1]<=target if direction=='min' else predictive_samples[:,-1]>=target
    probability=float(success.mean()) if sufficient else None
    # Whole-run holdout: models and weights use only earlier independent runs;
    # the test run's final observed value is never used in those fits.
    calibration=dict(status='not_established',reason='fewer than eight independent aligned-endpoint runs',calibrated_certificate=False)
    endpoints={r['points'][-1]['step'] for r in normalized}
    if n>=8 and len(endpoints)==1:
        order=np.random.default_rng(seed+1).permutation(n);test=order[-max(2,n//4):];train=order[:-len(test)]
        endpoint=next(iter(endpoints));predictions=[]
        for index in train:
            model=fits[int(index)][0]
            predictions.extend(sign*(_predict(model,[endpoint])[0]+np.array(model['residuals'])))
        lower,upper=np.quantile(predictions,[alpha,1-alpha]);actual=np.array([normalized[int(i)]['points'][-1]['value'] for i in test])
        calibration=dict(status='held_out_diagnostic',holdout_run_ids=[normalized[int(i)]['run_id'] for i in test],
            training_run_ids=[normalized[int(i)]['run_id'] for i in train],observations=len(test),
            observed_band_coverage=float(((actual>=lower)&(actual<=upper)).mean()),nominal_band_level=band_level,
            prediction_interval=[float(lower),float(upper)],absolute_mean_error=float(abs(actual.mean()-np.mean(predictions))),
            calibrated_certificate=False,scope='independent-run endpoint diagnostic at observed budget; does not calibrate future extrapolation')
    return dict(result,status='fitted',target_probability=probability,
        probability_reason=None if sufficient else 'insufficient independent runs or trajectory variation for a probability estimate',
        target_probability_estimand='new independent run from the same declared population at forecast horizon, conditional on the selected trend families',
        mean_trend_band=bands(mean_samples),new_run_prediction_band=bands(predictive_samples),
        probability_monte_carlo_standard_error=None if probability is None else float(np.sqrt(probability*(1-probability)/bootstrap_samples)),
        selected_fits=[dict(run_id=run['run_id'],**{k:v for k,v in models[0].items() if k!='residuals'}) for run,models in zip(normalized,fits)],
        calibration=calibration,asymptotic_values=[sign*m[0]['intercept'] if m[0]['family'] in {'constant','exponential','power'} else None for m in fits])


def declare_stop_rules(rules, *, checkpoint,objective,unit,declared_before_step):
    identity=_identity(checkpoint,objective,unit);before=_finite(declared_before_step,'declaration step')
    normalized=[]
    for rule in rules:
        if not isinstance(rule,dict) or rule.get('kind') not in {'threshold','budget','stagnation'}:
            raise StrataError('unknown observed stop rule')
        row=dict(rule);kind=row['kind']
        if kind=='threshold':
            if row.get('operator') not in {'<=','>='}:raise StrataError('threshold operator must be <= or >=')
            row['value']=_finite(row.get('value'),'threshold');row['patience']=row.get('patience',1)
            if type(row['patience'])is not int or row['patience']<1:raise StrataError('patience must be positive')
        elif kind=='budget':row['step']=_finite(row.get('step'),'budget step')
        else:
            if row.get('direction','min') not in {'min','max'}:raise StrataError('invalid stagnation direction')
            row['window']=row.get('window',3);row['minimum_improvement']=_finite(row.get('minimum_improvement',0),'minimum improvement')
            if type(row['window'])is not int or row['window']<2 or row['minimum_improvement']<0:raise StrataError('invalid stagnation rule')
        normalized.append(row)
    if not normalized:raise StrataError('declare at least one observed stop rule')
    result=dict(**identity,rules=normalized,declared_before_step=before)
    return dict(result,declaration_hash=digest(result))


def evaluate_stop_rules(observations,declaration,*,store):
    """Evaluate frozen rules on hash-bound observed metrics, never forecasts.

    Every observation is stored JSON with checkpoint/objective/unit, step and
    value; its ``artifact_hash`` references a real blob in the supplied Store.
    """
    from .store import Store
    payload={k:v for k,v in declaration.items() if k!='declaration_hash'}
    if digest(payload)!=declaration.get('declaration_hash'):raise StrataError('stop rule declaration hash mismatch')
    identity=_identity(declaration.get('checkpoint'),declaration.get('objective'),declaration.get('unit'))
    canonical_declaration=declare_stop_rules(declaration['rules'],**identity,declared_before_step=declaration['declared_before_step'])
    if canonical_declaration!=declaration:raise StrataError('stop rule declaration is not canonical')
    owned=not isinstance(store,Store);backend=Store(store) if owned else store
    checked=[]
    try:
        for observation in observations:
            if observation.get('kind')!='measured':raise StrataError('stop rules accept observed measurements only; forecasts cannot trigger them')
            _match(observation,identity);key=observation.get('artifact_hash')
            if not isinstance(key,str) or not re.fullmatch(r'[0-9a-f]{64}',key):raise StrataError('observation needs an artifact hash')
            try:stored=json.loads(backend.read_blob(key))
            except (ValueError,TypeError) as exc:raise StrataError('observation blob must contain JSON metrics') from exc
            required=dict(**identity,step=_finite(observation.get('step'),'step'),value=_finite(observation.get('value'),'value'))
            if not isinstance(stored,dict) or any(stored.get(k)!=v for k,v in required.items()):raise StrataError('observation artifact does not bind supplied metrics')
            checked.append(dict(required,artifact_hash=key))
    finally:
        if owned:backend.db.close()
    if not checked or any(b['step']<=a['step'] for a,b in zip(checked,checked[1:])):raise StrataError('observations must be nonempty and strictly increasing')
    if declaration['declared_before_step']>=checked[0]['step']:raise StrataError('stop rules were not declared before the observed series')
    outcomes=[]
    for rule in declaration['rules']:
        kind=rule['kind'];triggered=False
        if kind=='budget':triggered=checked[-1]['step']>=rule['step']
        elif kind=='threshold':
            tail=checked[-rule['patience']:]
            triggered=len(tail)==rule['patience'] and all(r['value']<=rule['value'] if rule['operator']=='<=' else r['value']>=rule['value'] for r in tail)
        elif len(checked)>=rule['window']:
            tail=checked[-rule['window']:];sign=1 if rule.get('direction','min')=='min' else -1
            improvement=sign*tail[0]['value']-min(sign*r['value'] for r in tail)
            triggered=improvement<=rule['minimum_improvement']
        outcomes.append(dict(rule=rule,triggered=bool(triggered)))
    return dict(**identity,kind='model',evidence_kind='model',model_kind='observed_stop_rule',stop=any(r['triggered'] for r in outcomes),
        declaration_hash=declaration['declaration_hash'],source_hash=digest(checked),observations=checked,rules=outcomes,
        scope='predeclared rules evaluated on retained observed metric artifacts; no forecast used',forecast_used=False)


def _fraction(value):
    if isinstance(value,bool):raise StrataError('rational model coefficients must be numeric')
    try:return value if isinstance(value,Fraction) else Fraction(value) if type(value)is int else Fraction(str(value))
    except (ValueError,ZeroDivisionError) as exc:
        # Re-read our own large exact strings without a process-wide override.
        if isinstance(value,str) and re.fullmatch(r'\s*[+-]?[0-9]+(?:/[0-9]+)?\s*',value):
            numerator,_,denominator=value.strip().partition('/')
            try:return Fraction(int(Decimal(numerator)),int(Decimal(denominator or '1')))
            except (ValueError,ZeroDivisionError):pass
        raise StrataError('invalid rational model coefficient') from exc


def _rational_text(value):
    """Format allowed exact coefficients without changing Python's digit limit."""
    try:return str(value)
    except ValueError:
        numerator=format(Decimal(value.numerator),'f')
        return numerator if value.denominator==1 else numerator+'/'+format(Decimal(value.denominator),'f')


def _quadratic_inputs(hessian,target,continuous,offset):
    if not isinstance(hessian,(list,tuple,np.ndarray)) or len(hessian)==0:raise StrataError('quadratic model needs a nonempty matrix')
    n=len(hessian)
    if any(not isinstance(row,(list,tuple,np.ndarray)) or len(row)!=n for row in hessian):raise StrataError('quadratic Hessian must be square')
    H=[[_fraction(v) for v in row] for row in hessian]
    if any(H[i][j]!=H[j][i] for i in range(n) for j in range(n)):raise StrataError('quadratic matrix must be exactly symmetric')
    t=[Fraction(0)]*n if target is None else [_fraction(v) for v in target]
    c=tuple(continuous)
    if len(t)!=n or len(set(c))!=len(c) or any(type(i)is not int or not 0<=i<n for i in c):raise StrataError('invalid target or continuous coordinates')
    return H,t,sorted(c),_fraction(offset)


def quadratic_analysis(hessian,*,target=None,steps=None,continuous=(),offset=0,parameter_names=None):
    """Floating-point curvature/Schur sensitivity, explicitly noncertified."""
    H,t,c,base=_quadratic_inputs(hessian,target,continuous,offset);n=len(H);q=[i for i in range(n) if i not in c]
    matrix=np.array(H,dtype=float);eigen=np.linalg.eigvalsh(matrix);scale=max(1.,float(np.linalg.norm(matrix,2)))
    if eigen[0]<-1e-10*scale:raise StrataError('quadratic matrix is not numerically positive semidefinite')
    step=np.zeros(n)
    supplied=[1.]*len(q) if steps is None else list(steps)
    if len(supplied)==n:
        # Continuous coordinates have no quantization grid. Their slots may
        # contain None/zero; normalize them away from the model identity.
        for i in c:
            if supplied[i] is not None and _finite(supplied[i],'continuous step')<0:
                raise StrataError('continuous grid placeholders must be nonnegative or None')
        supplied=[supplied[i] for i in q]
    elif len(supplied)!=len(q):
        raise StrataError('grid steps must match dimension or quantized coordinates')
    for i,value in zip(q,supplied):
        step[i]=_finite(value,'grid step')
        if step[i]<=0:raise StrataError('quantized grid steps must be positive')
    names=[str(i) for i in range(n)] if parameter_names is None else list(parameter_names)
    if len(names)!=n or any(not isinstance(name,str) for name in names):raise StrataError('parameter names must match dimension')
    block=matrix[np.ix_(q,q)];compensator=np.zeros((len(c),len(q)))
    if c:
        compensator=np.linalg.pinv(matrix[np.ix_(c,c)],rcond=1e-12)@matrix[np.ix_(c,q)]
        block=block-matrix[np.ix_(q,c)]@compensator
    block=(block+block.T)/2
    ranks=[]
    for index,i in enumerate(q):
        others=[j for j in range(len(q)) if j!=index];effective=float(block[index,index])
        if others:effective-=float(block[index,others]@np.linalg.pinv(block[np.ix_(others,others)],rcond=1e-12)@block[others,index])
        ranks.append(dict(index=i,name=names[i],raw_curvature=float(matrix[i,i]),grid_step=float(step[i]),
            raw_scale_score=float(matrix[i,i]*step[i]**2),compensated_curvature=float(block[index,index]),
            compensated_scale_score=float(block[index,index]*step[i]**2),fully_compensated_coordinate_curvature=max(0.,effective)))
    ranks.sort(key=lambda row:(-row['compensated_scale_score'],row['index']))
    coupling=[]
    for i in range(n):
        for j in range(i+1,n):
            if matrix[i,j]:coupling.append(dict(left=i,right=j,value=float(matrix[i,j]),
                normalized=float(matrix[i,j]/np.sqrt(matrix[i,i]*matrix[j,j])) if matrix[i,i]>0 and matrix[j,j]>0 else None))
    coupling.sort(key=lambda row:-abs(row['value']))
    model=dict(hessian=[[_rational_text(v) for v in row] for row in H],target=[_rational_text(v) for v in t],continuous=c,offset=_rational_text(base),
               steps=step.tolist(),parameter_names=names)
    return dict(kind='model',evidence_kind='model',model_kind='numerical_quadratic_model',model_hash=digest(model),source_hash=digest(model),
        formula='offset + (x-target)^T H (x-target)',dimension=n,quantized_indices=q,continuous_indices=c,
        eigenvalues=eigen.tolist(),schur_matrix=block.tolist(),schur_eigenvalues=np.linalg.eigvalsh(block).tolist() if q else [],
        continuous_compensator=compensator.tolist(),sensitivity=ranks,cross_coupling=coupling,
        numerical_bound_certified=False,scope='provided local PSD quadratic only; no bound on an unmodelled nonlinear pipeline',
        psd_check='floating-point eigenvalue tolerance; not an exact PSD certificate')


def _rational_psd(H,bit_budget):
    work=[row[:] for row in H];remaining=list(range(len(H)));rank=0
    while remaining:
        if any(work[i][i]<0 for i in remaining):raise StrataError('rational quadratic is not positive semidefinite')
        pivot=next((i for i in remaining if work[i][i]>0),None)
        if pivot is None:
            if any(work[i][j] for i in remaining for j in remaining):raise StrataError('rational quadratic is not positive semidefinite')
            break
        rest=[i for i in remaining if i!=pivot]
        for i in rest:
            for j in rest:
                work[i][j]-=work[i][pivot]*work[pivot][j]/work[pivot][pivot]
                if max(work[i][j].numerator.bit_length(),work[i][j].denominator.bit_length())>bit_budget:raise OverflowError('rational coefficient budget')
        remaining=rest;rank+=1
    return rank


def _solve_rational(matrix,rhs,bit_budget):
    n=len(matrix);columns=len(rhs[0]) if rhs else 0;rows=[matrix[i][:]+rhs[i][:] for i in range(n)];pivots=[];at=0
    for column in range(n):
        pivot=next((r for r in range(at,n) if rows[r][column]),None)
        if pivot is None:continue
        rows[at],rows[pivot]=rows[pivot],rows[at];value=rows[at][column];rows[at]=[v/value for v in rows[at]]
        for r in range(n):
            if r!=at:
                value=rows[r][column];rows[r]=[a-value*b for a,b in zip(rows[r],rows[at])]
        if any(max(v.numerator.bit_length(),v.denominator.bit_length())>bit_budget for row in rows for v in row):raise OverflowError('rational solve coefficient budget')
        pivots.append(column);at+=1
    if any(any(row[n:]) for row in rows[at:]):raise StrataError('continuous compensation is inconsistent with PSD model')
    solution=[[Fraction(0)]*columns for _ in range(n)]
    for row,column in enumerate(pivots):solution[column]=rows[row][n:]
    return solution


def exact_grid_floor(hessian,*,target,grids,continuous=(),offset=0,max_states=65536,max_dimension=12,max_fraction_bits=4096):
    """Exact bounded rational grid minimum after unrestricted continuous compensation.

    ``grids`` maps each noncontinuous coordinate to its complete finite allowed
    values. Decimal floats mean their decimal spelling, not hidden binary bits.
    Oversized models return an explicitly noncertified numerical estimate.
    """
    if any(type(v)is not int or v<1 for v in (max_states,max_dimension,max_fraction_bits)):raise StrataError('quadratic budgets must be positive integers')
    H,t,c,base=_quadratic_inputs(hessian,target,continuous,offset);n=len(H);q=[i for i in range(n) if i not in c]
    if not isinstance(grids,dict):raise StrataError('grids must map quantized coordinates to finite value lists')
    normalized={}
    for key,values in grids.items():
        if type(key)is int:index=key
        elif isinstance(key,str) and key.isdecimal():index=int(key)
        else:raise StrataError('grid coordinates must be integer indices')
        if index in normalized or index not in q or not isinstance(values,(list,tuple)) or not values:raise StrataError('invalid finite quantization grid')
        normalized[index]=sorted(set(_fraction(v) for v in values))
    if set(normalized)!=set(q):raise StrataError('every noncontinuous coordinate needs a finite grid')
    count=prod(len(normalized[i]) for i in q)
    model=dict(hessian=[[_rational_text(v) for v in row] for row in H],target=[_rational_text(v) for v in t],continuous=c,offset=_rational_text(base),
               grids={str(i):[_rational_text(v) for v in normalized[i]] for i in q})
    provenance=dict(kind='model',model_hash=digest(model),source_hash=digest(model),model=model,
        scope='minimum of the supplied rational PSD quadratic on these finite grids with unrestricted continuous compensation; not a nonlinear pipeline floor',
        formula='offset + (x-target)^T H (x-target)',grid_states=count,
        limits=dict(max_states=max_states,max_dimension=max_dimension,max_fraction_bits=max_fraction_bits))
    def within_budget(values):
        return all(max(v.numerator.bit_length(),v.denominator.bit_length())<=max_fraction_bits for v in values)
    def fallback(reason):
        try:
            analysis=quadratic_analysis(hessian,target=target,continuous=c,offset=offset)
            distances=[min(float(v-t[i])**2 for v in normalized[i]) for i in q]
            spectral=max(0.,min(analysis['schur_eigenvalues'],default=0.))*sum(distances)
            by_index={row['index']:row for row in analysis['sensitivity']}
            coordinate=max((by_index[i]['fully_compensated_coordinate_curvature']*d for i,d in zip(q,distances)),default=0.)
            estimate=float(base)+max(spectral,coordinate)
            if not np.isfinite(estimate):raise OverflowError('nonfinite fallback')
        except (OverflowError,np.linalg.LinAlgError) as exc:
            return dict(provenance,status='bounded',evidence_kind='model',model_kind='numerical_quadratic_model',reason=reason,
                certified=False,numerical_lower_bound_estimate=None,numerical_fallback_reason=str(exc))
        return dict(provenance,status='bounded',evidence_kind='model',model_kind='numerical_quadratic_model',reason=reason,
            certified=False,numerical_lower_bound_estimate=estimate,
            bound_method='Schur-complement spectral and separately compensated coordinate estimates',analysis=analysis)
    if count>max_states or n>max_dimension:return fallback('grid or dimension budget')
    if not within_budget([v for row in H for v in row]+t+[base]+[v for values in normalized.values() for v in values]):
        return fallback('rational input coefficient budget')
    try:
        rank=_rational_psd(H,max_fraction_bits)
        Y=_solve_rational([[H[i][j] for j in c] for i in c],[[H[i][j] for j in q] for i in c],max_fraction_bits) if c else []
        S=[[H[i][j]-sum(H[i][cc]*Y[k][b] for k,cc in enumerate(c)) for b,j in enumerate(q)] for i in q]
        if not within_budget(v for row in S for v in row):return fallback('rational Schur coefficient budget')
    except OverflowError as exc:return fallback(str(exc))
    best=None;witness=None
    for values in product(*(normalized[i] for i in q)):
        delta=[v-t[i] for i,v in zip(q,values)];value=base+sum(delta[a]*S[a][b]*delta[b] for a in range(len(q)) for b in range(len(q)))
        if not within_budget([value]):return fallback('rational grid evaluation budget')
        if best is None or value<best:
            best=value;point=t[:]
            for i,v in zip(q,values):point[i]=v
            for k,i in enumerate(c):point[i]=t[i]-sum(Y[k][j]*delta[j] for j in range(len(q)))
            if not within_budget(point):return fallback('rational minimizer coefficient budget')
            witness=point
    def float_summary(value):
        try:
            result=float(value)
            return result if np.isfinite(result) else None
        except OverflowError:return None
    return dict(provenance,status='exact',evidence_kind='model',model_kind='exact_small_rational_model',certified=True,certificate_scope='supplied finite rational model only',
        minimum=_rational_text(best),minimum_float=float_summary(best),minimizer=[_rational_text(v) for v in witness],
        minimizer_float=[float_summary(v) for v in witness],float_summary_scope='rounded display only; exact rational strings are authoritative',rational_psd_rank=rank,
        schur_matrix=[[_rational_text(v) for v in row] for row in S],enumerated_states=count,
        method='exact rational PSD elimination, exact continuous stationary solve, exhaustive finite grid enumeration')
