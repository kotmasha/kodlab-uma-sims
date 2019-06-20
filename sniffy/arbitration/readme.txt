This directory contains UMA scripts, data, and plots for a "Sniffy" agent on
the interval/circle, with simple learned arbitration.

The conjunction of the motor agents' decision bits (marked id_toLT and id_toRT
in the script code) is an input bit (id_toF) characterizing a failure mode of
the motor system; this bit is an input bit to a new BUA, ARB, whose action is
to flip the decision bit of one (randomly chosen) of {RT,LT}.

* In the experiments we have run, 
  - all agents are qualitative agents;
  - environments are interval/circle with 20 "geographic" sensors

* Three types of experiment:
  - "short":  random walk training for 250 cycles, followed by full control 
              authority to all BUAs until t=2000 cycles. 
  - "long":   random walk for 500 cycles, total 2000 cycles.
  - "longer": random walk for 1000 cycles, total 2000 cycles.

* Experiments comprising
  - 1000 runs each, of each type, with complete recording of the motor BUAs
    internal state (using sniffy_AT_arb.py);
  - 10000 runs each, of type "short", recording only the experiment state
    (agent position, distance to the target, cycle counter...), using the
    reduced script sniffy_AT_arb_mini.py.

* Plot types:
  - plot_batch.py <dir_name>

    Plots the mean & std. div. (over all runs) of Sniffy's distance to the
    target as a function of time from the data in <dir_name>, for two separate
    populations of runs:
    - population A (red): runs *terminating* at dist < diam/2 from the target;
    - population B (blue): all other runs.

  - plot_batch_split.py <dir_name>

    Plots Sniffy's distance to the target as a function of time, splitting the
    data into two population as above.

  - plot_one.py <dir_name> <num>

    Plots full state information for the run <num> in <dir_name>.
