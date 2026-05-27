# CIRCA-RT Theory And Strong Claim

Updated: 2026-05-24

This note reframes CIRCA-RT as a real-time systems mechanism, not merely an
anomaly detector. The strongest paper angle is:

> CIRCA-RT turns an arbitrary semantic-timing risk score into a schedulable
> runtime audit service. Detection quality is empirical; audit demand and
> audit-induced deadline risk are bounded by design.

This is the part that can be made distinctive for RTSS. Most anomaly detectors
output a score. CIRCA-RT outputs a score plus a runtime contract:

```text
monitor cost m
heavy-audit charge c
bucket capacity B
replenishment rate r
deadline D
windowed audit-demand bound
deadline-miss augmentation bound
```

## 1. Model

For each frame `t`, let:

```text
X_t    baseline latency before CIRCA-RT audit
m      inline monitor cost
c      heavy-audit cost or audit token charge
D      frame deadline
s_t    semantic residual
d_t    timing residual
z_t    runtime context and mode
q_t    CIRCA-RT dependence score
a_t    heavy-audit decision in {0,1}
b_t    token-bucket level after frame t
B      bucket capacity
r      per-frame token replenishment
```

The implemented latency accounting is:

```text
L_t = X_t + m + c a_t.
```

The current implementation computes:

1. context-conditioned residuals:

```text
u_t = s_t - E_hat[s_t | z_t]
v_t = d_t - E_hat[d_t | z_t]
```

2. random Fourier features:

```text
phi_u(t) = phi(u_t)
phi_v(t) = psi(v_t)
```

3. a centered rolling cross-covariance score:

```text
q_t = || C_t - mean_phi_u,t mean_phi_v,t^T ||_F^2 / w_t^2
```

4. a two-threshold policy:

```text
alarm_t = 1 if q_t >= q_low
audit_t = 1 if q_t >= q_high and bucket has at least c tokens
```

The code corresponds to `src/circa_rt/core.py` and
`src/circa_rt/features.py`.

## 2. Theorem: Token-Bucket Audit Demand

**Theorem 1.** For any contiguous interval `I` with `|I| = N`, CIRCA-RT performs
at most:

```text
A(I) = sum_{t in I} a_t <= floor((B + rN) / c)
```

heavy audits, where `B` is bucket capacity, `r` is per-frame replenishment, and
`c` is the token charge per heavy audit.

**Proof.** Let `b_t` be the bucket level after frame `t`. At each frame,
CIRCA-RT first replenishes at most `r` tokens and caps the bucket at `B`. If an
audit is performed, it removes exactly `c` tokens. Therefore, for every frame:

```text
b_t <= b_{t-1} + r - c a_t.
```

Summing over any interval `I = {s, ..., e}` gives:

```text
b_e <= b_{s-1} + rN - c sum_{t in I} a_t.
```

Rearranging:

```text
c A(I) <= b_{s-1} + rN - b_e.
```

Because `0 <= b_e` and `b_{s-1} <= B`:

```text
c A(I) <= B + rN.
```

Since `A(I)` is integral:

```text
A(I) <= floor((B + rN) / c).
```

This bound is independent of the score distribution, attack distribution, and
baseline latency distribution.

## 3. Theorem: Audit-Induced Deadline-Miss Bound

The token-bucket theorem should be connected directly to deadline misses. This
is stronger than only reporting audit rate.

Define the monitor-only deadline miss indicator:

```text
M_t^0 = 1{ X_t + m > D }.
```

Define the CIRCA-RT deadline miss indicator:

```text
M_t = 1{ X_t + m + c a_t > D }.
```

**Theorem 2.** For any interval `I` with `|I| = N`, CIRCA-RT can introduce at
most `floor((B+rN)/c)` additional deadline misses beyond the monitor-only path:

```text
sum_{t in I} M_t <= sum_{t in I} M_t^0 + floor((B + rN) / c).
```

**Proof.** If `M_t = 1` and `M_t^0 = 0`, then:

```text
X_t + m <= D < X_t + m + c a_t.
```

This can occur only if `a_t = 1`. Therefore every audit-induced miss is charged
to one audited frame:

```text
sum_{t in I} (M_t - M_t^0)_+ <= sum_{t in I} a_t = A(I).
```

By Theorem 1:

```text
A(I) <= floor((B + rN) / c).
```

Thus:

```text
sum_{t in I} M_t <= sum_{t in I} M_t^0 + floor((B + rN) / c).
```

This theorem is useful because it gives a deterministic worst-case bound even
when heavy GPU interference makes `X_t` have a bad tail.

## 4. Proposed Upgrade: Slack-Admissible CIRCA-RT

The current implementation is token-bucket bounded, but it does not check
per-frame deadline slack before auditing. A stronger variant can be added:

```text
a_t = 1 only if
  q_t >= q_high,
  b_t >= c,
  X_t + m + c <= D - epsilon.
```

This is implemented as the optional method `CIRCA-RT-Slack`.

**Theorem 3.** Under slack-admissible auditing, CIRCA-RT introduces no additional
deadline misses beyond the monitor-only path:

```text
M_t = M_t^0 for all t.
```

**Proof.** If `M_t^0 = 1`, the monitor-only path already misses the deadline.
If `M_t^0 = 0`, then `X_t + m <= D`. Slack-admissible CIRCA-RT audits only when:

```text
X_t + m + c <= D - epsilon <= D.
```

Therefore an audited frame cannot turn a monitor-only non-miss into a miss.
Non-audited frames have latency `X_t + m`, so they also cannot introduce a new
miss. Thus `M_t = M_t^0` for every frame.

This proposed variant is a strong RTSS contribution: the system can separate
urgent warnings from heavy audits. The alarm can still fire when risk is high,
but the expensive audit is admitted only when the real-time slack allows it.

## 4.1. Theorem: Maximal Safe Audit Admission

The slack rule can be stated more strongly than "it is safe." Under a
per-frame deadline model, it is the maximal safe admission rule for heavy audit,
subject to the token budget.

Let a high-risk frame be one with:

```text
q_t >= q_high.
```

Let the monitor-only latency be:

```text
L_t^0 = X_t + m.
```

Let the audited latency be:

```text
L_t^1 = X_t + m + c.
```

Assume the audit cost bound `c` is conservative for the heavy audit action used
by the admission controller.

**Theorem 3a.** For any high-risk frame `t` whose token bucket has at least `c`
tokens, the slack-admissible policy admits the audit if and only if executing the
audit is locally schedulable:

```text
a_t = 1  <=>  L_t^1 <= D - epsilon.
```

Consequently, among all policies that may only use the current frame's measured
latency, deadline, audit-cost bound, score, and token state, CIRCA-RT-Slack
admits every safe high-risk audit and rejects every high-risk audit that would
violate the per-frame deadline.

**Proof.** For a high-risk frame with sufficient tokens, the implemented slack
predicate is exactly:

```text
X_t + m + c <= D - epsilon.
```

This predicate is identical to local schedulability of the audited frame under
the conservative audit-cost bound. If the predicate is true, executing the audit
cannot violate the deadline, so rejecting it would be unnecessarily conservative
under the local model. If the predicate is false, then:

```text
X_t + m + c > D - epsilon.
```

With `epsilon = 0`, executing the audit violates the deadline. With
`epsilon > 0`, executing it violates the configured safety margin. Therefore no
locally deadline-safe policy can admit that audit under the same cost bound and
margin. Token availability is an independent long-horizon budget constraint, so
the maximality claim is conditional on having sufficient tokens.

This theorem gives the method a useful systems interpretation:

```text
score policy:      which frames are risky?
token policy:      how much heavy audit demand can the system afford over time?
slack policy:      which risky audits are locally schedulable right now?
```

The latest `results_deadline_safe_admission_20260524` experiment supports this
claim empirically: wrapping external monitor scores with the slack admission
layer preserves alarm recall while reducing audit-induced deadline misses to
zero under quantile-relative tight deadlines.

## 5. Theorem: Context-Residual Dependence Separation

CIRCA-RT should not be presented as a raw anomaly detector. Its score is a
conditional dependence monitor.

Assume benign operation has:

```text
s_t = f_s(z_t) + eps_s,t
d_t = f_d(z_t) + eps_d,t
eps_s,t independent of eps_d,t conditional on z_t.
```

If the residualizer estimates `f_s` and `f_d` exactly, then:

```text
u_t = eps_s,t
v_t = eps_d,t
```

and `u_t` and `v_t` are independent under benign operation.

**Theorem 4.** Under the benign additive-context model above, the population
HSIC between `u_t` and `v_t` is zero for a characteristic kernel. Under a
coupled semantic-timing attack of the form:

```text
s_t = f_s(z_t) + alpha h_t + eps_s,t
d_t = f_d(z_t) + beta  h_t + eps_d,t
```

with non-degenerate shared factor `h_t` and `alpha beta != 0`, the population
HSIC is strictly positive for characteristic kernels.

**Proof sketch.** Under benign operation, exact residualization gives
`u_t = eps_s,t` and `v_t = eps_d,t`. Conditional independence plus the additive
model removes the shared context term, so the residual joint distribution
factorizes:

```text
P_{u,v} = P_u P_v.
```

For characteristic kernels, HSIC equals zero if and only if the joint
distribution factorizes, so benign HSIC is zero.

Under the coupled attack, both residuals contain the same non-degenerate latent
factor `h_t`:

```text
u_t = alpha h_t + eps_s,t
v_t = beta  h_t + eps_d,t.
```

When `alpha beta != 0`, the joint distribution does not factorize except in
degenerate cases. Characteristic-kernel HSIC is therefore strictly positive.

This theorem explains why simple semantic-only or timing-only thresholds are not
the right baseline for the hardest attack. CIRCA-RT is designed for dependence
that appears after context has been removed.

## 6. Theorem: Online Cost Is Bounded

Let `p` be the encoded context dimension and `d` be the RFF feature dimension.
The online work per frame is:

```text
residualization:       O(p)
RFF transforms:        O(d)
rolling cross update:  O(d^2)
token bucket update:   O(1)
```

The memory is:

```text
O(wd + d^2)
```

where `w` is the rolling window size.

In the current configuration, `d = 32` and `w = 32`, so the monitor path is
constant-time with respect to the total trace length. This supports the paper's
claim that CIRCA-RT is a runtime monitor rather than an offline batch detector.

## 7. Score-Policy Decoupling Proposition

`ConditionalRFF-HSIC` and CIRCA-RT can share nearly the same score construction:

```text
conditional residualization
random Fourier features
rolling cross-covariance score
benign quantile threshold
```

The difference is policy:

```text
ConditionalRFF-HSIC: score -> alarm/audit directly
CIRCA-RT:            score -> alarm -> token-bucket audit admission
```

**Proposition.** Under the same residualizer, RFF seeds, window size, and
calibration split, CIRCA-RT and ConditionalRFF-HSIC induce the same score
ordering. Any difference in audit workload comes from audit scheduling, not from
a different anomaly score.

This is not a weakness. It makes the paper cleaner: CIRCA-RT is a scheduling
mechanism that can wrap a conditional dependence score. The contribution is the
real-time audit contract, not claiming a completely new statistical test.

## 8. Strongest Paper Positioning

Weak positioning:

> CIRCA-RT is a better anomaly detector.

Stronger RTSS positioning:

> CIRCA-RT is a runtime audit server for real-time AI pipelines. It converts
> semantic-timing risk scores into bounded heavy-audit demand and bounded
> audit-induced deadline risk.

Potentially disruptive positioning:

> Real-time ML monitors should expose a demand contract, not only accuracy
> metrics. CIRCA-RT is a first step toward contract-based runtime auditing:
> empirical detection quality plus deterministic audit-demand bounds.

This lets the paper argue that existing runtime monitors are incomplete for
real-time systems because they usually report AUROC/recall but not the
worst-case cost of acting on their own alarms.

## 9. What To Add Next

Implemented optional method:

```text
CIRCA-RT-Slack
```

It audits only if score is high, a token is available, and measured slack admits
the audit cost. The default experiment method list is unchanged; this method is
enabled only when explicitly requested.

Expected benefits:

- turns Theorem 3 into an implemented result
- directly addresses T4 heavy-tail deadline failures
- gives a sharper RTSS claim: alerts remain online, but heavy audits are
  admitted only when schedulable
- makes AlwaysAudit look structurally unsafe under tight deadlines

Required experiment table:

```text
Policy                 Recall   Audit rate   Audit-induced misses   Total misses   p99
AlwaysAudit
CIRCA-RT
Slack-Admissible CIRCA-RT
ContextAwareConformal
```

Most important metric:

```text
additional_miss_ratio = miss_ratio(policy) - miss_ratio(NoAuditMonitor)
```

If Slack-Admissible CIRCA-RT has near-zero additional misses while preserving
useful recall, that is the strongest new result.

Local smoke check on a synthetic tight-deadline trace:

```text
regular CIRCA-RT:     36 audits, miss ratio 0.3875
CIRCA-RT-Slack:       13 audits, miss ratio 0.3500
extra slack misses:   0
```

## 10. Honest Limitations

These proofs do not prove high detection recall. They prove resource behavior.

What remains empirical:

- attack recall
- false alarm rate
- score separation
- latency distribution under GPU interference
- whether TensorRT/Jetson tails stay below a given deadline

What is deterministic:

- maximum audit count in any window
- maximum audit work admitted in any window
- maximum number of audit-induced deadline misses
- zero audit-induced misses under the proposed slack-admissible variant
- online monitor complexity for fixed `d` and `w`

This separation is exactly the RTSS-friendly story.
