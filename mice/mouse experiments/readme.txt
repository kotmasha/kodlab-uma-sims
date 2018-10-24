This directory contains the results of the first extensive batch of experiments in the "mice" simulation environmnet.

Experimentation was conducted in four modes of the UMA agents, all subject to pre-wired arbitration (see tech. report Sections 4.1.2, 4.2.2 for detailed description).
    "random" mode:  the mouse performs randomized actions (the mouse_rand*.png files);
    "uniform" mode: the mouse learns implications based on relative frequencies, and explores based on specified target sensations (the mouse_unival*.png files);
    "real" mode:    the mouse learns implications based on motivational signals encoding the task of following the gradient of the "scent plume" emitted by the "cheese" targets (the mouse_realval*.png files);
    "auto" mode:    the mouse learns implications AND calculates its target state at each cycle, based on the motivational signals (the mouse_auto*.png files).
    
Run parameters are:
    learning discount parameter:    $q=(1-2^{-Q})$ for $Q=5,6,7,8,9$;
    step/turn bias parameter:       $k=k*10%$ for $K=4,5,6,7,8$;
    Training arena size:            $20*20$;
    Arena size:                     N-by-N for $N=40,60,100$;
    Sensory window size:              11-by-11
    Density of cheeses:             Uniform, with 0.75 expected number of cheeses per window;
    Time to pick-up:                A cheese is "eaten" if the agent stays within 3 steps from it for 50 consecutive cycles.    
    
Each combination of parameters was run 25 times; the plots represent means (thick curves) and mean +/- 1 sigma (thin curves).
    Top Left:   number of remaining cheeses;
    Top Right:  percentage of deliberate actions taken over time;
    Bot Left:   distance to nearest cheese;
    Bot Right:  discounted (using the parameter $q$) accumulated "reward" (each eaten cheese produces one unit of reward, which grows "stale" over time).

Side-by-side comparisons of the four modes of operation are provided in the files mouse_aggregate*.png, and discussed in the tech report, Section 4.2.2.
