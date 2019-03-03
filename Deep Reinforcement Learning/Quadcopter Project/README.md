<div tabindex="-1" id="notebook" class="border-box-sizing">

<div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html"># Project: Train a Quadcopter How to Fly[¶](#Project:-Train-a-Quadcopter-How-to-Fly) Design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! Try to apply the techniques you have learnt, but also feel free to come up with innovative ideas and test them.</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">## Instructions[¶](#Instructions) Take a look at the files in the directory to better understand the structure of the project. * `task.py`: Define your task (environment) in this file. * `agents/`: Folder containing reinforcement learning agents. * `policy_search.py`: A sample agent has been provided here. * `agent.py`: Develop your agent here. * `physics_sim.py`: This file contains the simulator for the quadcopter. **DO NOT MODIFY THIS FILE**. For this project, you will define your own task in `task.py`. Although we have provided a example task to get you started, you are encouraged to change it. Later in this notebook, you will learn more about how to amend this file. You will also design a reinforcement learning agent in `agent.py` to complete your chosen task. You are welcome to create any additional files to help you to organize your code. For instance, you may find it useful to define a `model.py` file defining any needed neural network architectures. ## Controlling the Quadcopter[¶](#Controlling-the-Quadcopter) We provide a sample agent in the code cell below to show you how to use the sim to control the quadcopter. This agent is even simpler than the sample agent that you'll examine (in `agents/policy_search.py`) later in this notebook! The agent controls the quadcopter by setting the revolutions per second on each of its four rotors. The provided agent in the `Basic_Agent` class below always selects a random action for each of the four rotors. These four speeds are returned by the `act` method as a list of four floating-point numbers. For this project, the agent that you will implement in `agents/agent.py` will have a far more intelligent method for selecting actions!</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [40]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">random</span>

<span class="k">class</span> <span class="nc">Basic_Agent</span><span class="p">():</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">task</span> <span class="o">=</span> <span class="n">task</span>

    <span class="k">def</span> <span class="nf">act</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="n">new_thrust</span> <span class="o">=</span> <span class="n">random</span><span class="o">.</span><span class="n">gauss</span><span class="p">(</span><span class="mf">450.</span><span class="p">,</span> <span class="mf">25.</span><span class="p">)</span>
        <span class="k">return</span> <span class="p">[</span><span class="n">new_thrust</span> <span class="o">+</span> <span class="n">random</span><span class="o">.</span><span class="n">gauss</span><span class="p">(</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">1.</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">4</span><span class="p">)]</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">Run the code cell below to have the agent select actions to control the quadcopter. Feel free to change the provided values of `runtime`, `init_pose`, `init_velocities`, and `init_angle_velocities` below to change the starting conditions of the quadcopter. The `labels` list below annotates statistics that are saved while running the simulation. All of this information is saved in a text file `data.txt` and stored in the dictionary `results`.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [41]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="o">%</span><span class="k">load_ext</span> autoreload
<span class="o">%</span><span class="k">autoreload</span> 2

<span class="kn">import</span> <span class="nn">csv</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">task</span> <span class="k">import</span> <span class="n">Task</span>

<span class="c1"># Modify the values below to give the quadcopter a different starting position.</span>
<span class="n">runtime</span> <span class="o">=</span> <span class="mf">10.</span>                                     <span class="c1"># time limit of the episode</span>
<span class="n">init_pose</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span> <span class="mf">10.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span> <span class="mf">4.</span><span class="p">])</span>  <span class="c1"># initial pose</span>
<span class="n">init_velocities</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">])</span>         <span class="c1"># initial velocities</span>
<span class="n">init_angle_velocities</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">])</span>   <span class="c1"># initial angle velocities</span>
<span class="n">file_output</span> <span class="o">=</span> <span class="s1">'data.txt'</span>                         <span class="c1"># file name for saved results</span>

<span class="c1"># Setup</span>
<span class="n">task</span> <span class="o">=</span> <span class="n">Task</span><span class="p">(</span><span class="n">init_pose</span><span class="p">,</span> <span class="n">init_velocities</span><span class="p">,</span> <span class="n">init_angle_velocities</span><span class="p">,</span> <span class="n">runtime</span><span class="p">)</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">Basic_Agent</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>
<span class="n">done</span> <span class="o">=</span> <span class="kc">False</span>
<span class="n">labels</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'time'</span><span class="p">,</span> <span class="s1">'x'</span><span class="p">,</span> <span class="s1">'y'</span><span class="p">,</span> <span class="s1">'z'</span><span class="p">,</span> <span class="s1">'phi'</span><span class="p">,</span> <span class="s1">'theta'</span><span class="p">,</span> <span class="s1">'psi'</span><span class="p">,</span> <span class="s1">'x_velocity'</span><span class="p">,</span>
          <span class="s1">'y_velocity'</span><span class="p">,</span> <span class="s1">'z_velocity'</span><span class="p">,</span> <span class="s1">'phi_velocity'</span><span class="p">,</span> <span class="s1">'theta_velocity'</span><span class="p">,</span>
          <span class="s1">'psi_velocity'</span><span class="p">,</span> <span class="s1">'rotor_speed1'</span><span class="p">,</span> <span class="s1">'rotor_speed2'</span><span class="p">,</span> <span class="s1">'rotor_speed3'</span><span class="p">,</span> <span class="s1">'rotor_speed4'</span><span class="p">]</span>
<span class="n">results</span> <span class="o">=</span> <span class="p">{</span><span class="n">x</span> <span class="p">:</span> <span class="p">[]</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">labels</span><span class="p">}</span>

<span class="c1"># Run the simulation, and save the results.</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">file_output</span><span class="p">,</span> <span class="s1">'w'</span><span class="p">)</span> <span class="k">as</span> <span class="n">csvfile</span><span class="p">:</span>
    <span class="n">writer</span> <span class="o">=</span> <span class="n">csv</span><span class="o">.</span><span class="n">writer</span><span class="p">(</span><span class="n">csvfile</span><span class="p">)</span>
    <span class="n">writer</span><span class="o">.</span><span class="n">writerow</span><span class="p">(</span><span class="n">labels</span><span class="p">)</span>
    <span class="k">while</span> <span class="kc">True</span><span class="p">:</span>
        <span class="n">rotor_speeds</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">act</span><span class="p">()</span>
        <span class="n">_</span><span class="p">,</span> <span class="n">_</span><span class="p">,</span> <span class="n">done</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">rotor_speeds</span><span class="p">)</span>
        <span class="n">to_write</span> <span class="o">=</span> <span class="p">[</span><span class="n">task</span><span class="o">.</span><span class="n">sim</span><span class="o">.</span><span class="n">time</span><span class="p">]</span> <span class="o">+</span> <span class="nb">list</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">sim</span><span class="o">.</span><span class="n">pose</span><span class="p">)</span> <span class="o">+</span> <span class="nb">list</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">sim</span><span class="o">.</span><span class="n">v</span><span class="p">)</span> <span class="o">+</span> <span class="nb">list</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">sim</span><span class="o">.</span><span class="n">angular_v</span><span class="p">)</span> <span class="o">+</span> <span class="nb">list</span><span class="p">(</span><span class="n">rotor_speeds</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">ii</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">labels</span><span class="p">)):</span>
            <span class="n">results</span><span class="p">[</span><span class="n">labels</span><span class="p">[</span><span class="n">ii</span><span class="p">]]</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">to_write</span><span class="p">[</span><span class="n">ii</span><span class="p">])</span>
        <span class="n">writer</span><span class="o">.</span><span class="n">writerow</span><span class="p">(</span><span class="n">to_write</span><span class="p">)</span>
        <span class="k">if</span> <span class="n">done</span><span class="p">:</span>
            <span class="k">break</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>The autoreload extension is already loaded. To reload it, use:
  %reload_ext autoreload
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">Run the code cell below to visualize how the position of the quadcopter evolved during the simulation.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [42]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'x'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'x'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'y'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'y'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'z'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'z'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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 )</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">The next code cell visualizes the velocity of the quadcopter.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [43]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'x_velocity'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'x_hat'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'y_velocity'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'y_hat'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'z_velocity'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'z_hat'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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 )</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">Next, you can plot the Euler angles (the rotation of the quadcopter over the $x$-, $y$-, and $z$-axes),</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [44]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'phi'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'phi'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'theta'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'theta'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'psi'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'psi'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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 )</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">before plotting the velocities (in radians per second) corresponding to each of the Euler angles.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [45]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'phi_velocity'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'phi_velocity'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'theta_velocity'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'theta_velocity'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'psi_velocity'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'psi_velocity'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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 )</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">Finally, you can use the code cell below to print the agent's choice of actions.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [46]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'rotor_speed1'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Rotor 1 revolutions / second'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'rotor_speed2'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Rotor 2 revolutions / second'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'rotor_speed3'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Rotor 3 revolutions / second'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">results</span><span class="p">[</span><span class="s1">'time'</span><span class="p">],</span> <span class="n">results</span><span class="p">[</span><span class="s1">'rotor_speed4'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Rotor 4 revolutions / second'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">_</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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8+c//5mNGzdyxx138NBDDw1pH2MNHegNdT5ff/opfOQjHxmz9xOGwf88IGWKD2X0Cr7vf9T3/Rm+788BLgXu9n3/MuAPwEVqsyuAX6p//0r9jXr+bv9AmMoCPKVkmZpISt4hHDwIBQlziEJ54yVTLIQI5IALhQKFQqFicFu7di0f+9jHOPnkk/nqV7/K7t27edOb3sTKlStZuXIlDzzwAJ7nMWfOnJKVyOGHH86uXbvYunUrp512GkuWLOG0007jpZdeqvgeejByz549zJkzh3w+z/XXX8/69etZfcLJ/OSOO/jRz34ZmG9U2++VV17JtddeywknnMC8efMC+eNqUsOlEDzx7LMsP+qoku/w6aefZtWqVSxbtowlS5bwwgsvAPCDH/wgePxv//Zvg8nVO+64g2OOOYalS5dy2mmnAUOXVQb49Kc/zZFHHsnpp5/Oc889V+F4R46RDExdB9wuhLgJeBz4d/X4vwPfF0JsRmbyl47sEEcPLxx5NP6rf+T0tkOB/v8yPve/n+PZfc+OeD9OPodRkNnvwr7lfOT4jzb0uvGWKXZdlxUrVrB582auueaaqjLFXV1d/PGPfwTgrW99K//wD//ASSedxEsvvcSZZ57Jpk2bOO+88/j5z3/OVVddxcMPP8ycOXOYPHky55xzDm9/+9u54oor+Pa3v821117LL37xi7rnIhKJ8KlPfYpHH32UL3zxc7D5r3zrt78Mnn/ve99bdb87duzg/vvv59lnn+Xcc8/loosuqig1PBhPPPUMi488EiFECdXym9/8Ju9///u57LLLyOfzuK7Lpk2bWL9+PQ888AC2bfOe97yHH/7wh7zhDW/gne98J/feey9z585l3z5ZtBiqrPITTzzB7bffzuOPP47jOBxzzDGsWLGi7nkbKoYU6H3fvwe4R/37RWBVhW2ywMWjcGyjjvTKi/lc1ODNyUOsm0MYBYQT4wbMrLVM8ZYtW1ixYkVdmeLB4l33338/73vf+4ChyRSbpsnGjRvp6uriggsu4KmnnmLRokVl24Ulcu+8806eeeaZ4O+enh56e3t585vfzKc+9Smuuuoqbr/99uA1Dz74ID/72c8AuPzyy/nwhz9c93wMhmEYku0ROpW19nv++edjGAZHHXUUu3btAqTU8NVXX02hUOD8888P9H3CuOsP97DupJOAUk796tWr+fSnP822bdu48MILOeKII7jrrrvYsGFDIEecyWTo7OzkoYceYs2aNcydK3uA+tzff//9/PSnPwWqyypHo9FAVvm+++7jggsuIJGQQ5znnnvukM9bIzhoJRAq4X0nncylS48nesh45P80rlt13ajsp2fPDuydewGIVwicgzFRMsUaLS0trF27ljvuuKNioA/vw/M8HnzwQeLxUrmQ1atXs3nzZnbv3s0vfvELPvGJT1R8r0rlobBMcTabLXveMExyZpEdV2+/YbEwfW4qSQ2//e1vL9nH3ff8kXd+VjV7Qzfot771rRx33HH85je/4cwzz+Rb3/oWvu9zxRVX8NnPfrZkH7/61a8alimuJas8+DONFQ6slvMYI2abzGw7JH9wCKMD3YwdKh1wPGWKd+/eHdTUM5kMd955JwsWLKh7jOvWrePrX/968LcuPwghuOCCC/jABz7AwoULaW+XzOkTTjiB22+/HYAf/vCHnKQy5jDCMsVhS0H9OQ3D5OUOgRsvBsRG9htGJanhMLq7u3Edl/aWFvV5iiHwxRdfZN68eVx77bWce+65PPHEE5x22mn85Cc/4dVXXwVkDX7r1q2sXr2aP/7xj/z1r38NHoehyyqvWbOGn//852QyGXp7e/n1r39d8/MNF/+nMvpDOITRRMD7HsZrx0umeMeOHVxxxRW4rovneVxyySWcffbZdY/va1/7Gtdccw1LliwJdNa/+c1vArLEs3LlSr7zne+UbH/11VfzhS98gY6ODm677bayfX7wgx/kkksu4fvf/z6nnnpq8Pgpp5zCzTffzIpjVvC2a96GbdhD2m8YlaSGw/if//kfTj31lODvMHNq/fr1/OAHP8C2baZMmcL1119PW1sbN910E+vWrcPzPGzb5pZbbuH444/n1ltv5cILL8TzPDo7O/n/27vXGKnuOozj34dl2+XSUpClIkugBCUgicslWEWIATStNtQXGmhCrZexJl5S6gsjxER9AykxjSSaJs1CqVK5FEqyAURIKtESxXKp0spqsCId2gqltNtqKbefL85/YXbZy5nZnf2fmf19kg3DMGd4FnZ/e+Z/znlm3759Rdcqz5w5kyVLltDY2MiECROYN29et48vVdXWFDtXqBw1xf9rPY9OnebKIBg+reelG9c9M6PlzRbqh9Yzekh5ToHO5XI88OX7mT0qeSViE8czdHjEBtQieE2x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 )</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">When specifying a task, you will derive the environment state from the simulator. Run the code cell below to print the values of the following variables at the end of the simulation: * `task.sim.pose` (the position of the quadcopter in ($x,y,z$) dimensions and the Euler angles), * `task.sim.v` (the velocity of the quadcopter in ($x,y,z$) dimensions), and * `task.sim.angular_v` (radians/second for each of the three Euler angles).</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [47]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1"># the pose, velocity, and angular velocity of the quadcopter at the end of the episode</span>
<span class="nb">print</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">sim</span><span class="o">.</span><span class="n">pose</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">sim</span><span class="o">.</span><span class="n">v</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">sim</span><span class="o">.</span><span class="n">angular_v</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>[ 69.27132741 -66.97697587   0\.           5.28708504   6.17506783   4\.        ]
[ 60.70042743 -48.63863183 -58.39179543]
[-0.12994726  0.03219731  0\.        ]
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">In the sample task in `task.py`, we use the 6-dimensional pose of the quadcopter to construct the state of the environment at each timestep. However, when amending the task for your purposes, you are welcome to expand the size of the state vector by including the velocity information. You can use any combination of the pose, velocity, and angular velocity - feel free to tinker here, and construct the state to suit your task. ## The Task[¶](#The-Task) A sample task has been provided for you in `task.py`. Open this file in a new window now. The `__init__()` method is used to initialize several variables that are needed to specify the task. * The simulator is initialized as an instance of the `PhysicsSim` class (from `physics_sim.py`). * Inspired by the methodology in the original DDPG paper, we make use of action repeats. For each timestep of the agent, we step the simulation `action_repeats` timesteps. If you are not familiar with action repeats, please read the **Results** section in [the DDPG paper](https://arxiv.org/abs/1509.02971). * We set the number of elements in the state vector. For the sample task, we only work with the 6-dimensional pose information. To set the size of the state (`state_size`), we must take action repeats into account. * The environment will always have a 4-dimensional action space, with one entry for each rotor (`action_size=4`). You can set the minimum (`action_low`) and maximum (`action_high`) values of each entry here. * The sample task in this provided file is for the agent to reach a target position. We specify that target position as a variable. The `reset()` method resets the simulator. The agent should call this method every time the episode ends. You can see an example of this in the code cell below. The `step()` method is perhaps the most important. It accepts the agent's choice of action `rotor_speeds`, which is used to prepare the next state to pass on to the agent. Then, the reward is computed from `get_reward()`. The episode is considered done if the time limit has been exceeded, or the quadcopter has travelled outside of the bounds of the simulation. In the next section, you will learn how to test the performance of an agent on this task.</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">## The Agent[¶](#The-Agent) The sample agent given in `agents/policy_search.py` uses a very simplistic linear policy to directly compute the action vector as a dot product of the state vector and a matrix of weights. Then, it randomly perturbs the parameters by adding some Gaussian noise, to produce a different policy. Based on the average reward obtained in each episode (`score`), it keeps track of the best set of parameters found so far, how the score is changing, and accordingly tweaks a scaling factor to widen or tighten the noise. Run the code cell below to see how the agent performs on the sample task.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [48]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">sys</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">from</span> <span class="nn">agents.policy_search</span> <span class="k">import</span> <span class="n">PolicySearch_Agent</span>
<span class="kn">from</span> <span class="nn">task</span> <span class="k">import</span> <span class="n">Task</span>

<span class="n">num_episodes</span> <span class="o">=</span> <span class="mi">1000</span>
<span class="n">target_pos</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span><span class="mf">58.</span><span class="p">])</span>
<span class="n">task</span> <span class="o">=</span> <span class="n">Task</span><span class="p">(</span><span class="n">target_pos</span><span class="o">=</span><span class="n">target_pos</span><span class="p">)</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">PolicySearch_Agent</span><span class="p">(</span><span class="n">task</span><span class="p">)</span> 

<span class="k">for</span> <span class="n">i_episode</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">num_episodes</span><span class="o">+</span><span class="mi">1</span><span class="p">):</span>
    <span class="n">state</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">reset_episode</span><span class="p">()</span> <span class="c1"># start a new episode</span>
    <span class="k">while</span> <span class="kc">True</span><span class="p">:</span>
        <span class="n">action</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">act</span><span class="p">(</span><span class="n">state</span><span class="p">)</span> 
        <span class="n">next_state</span><span class="p">,</span> <span class="n">reward</span><span class="p">,</span> <span class="n">done</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>
        <span class="n">agent</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">reward</span><span class="p">,</span> <span class="n">done</span><span class="p">)</span>
        <span class="n">state</span> <span class="o">=</span> <span class="n">next_state</span>
        <span class="k">if</span> <span class="n">done</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="s2">"</span><span class="se">\r</span><span class="s2">Episode =</span> <span class="si">{:4d}</span><span class="s2">, score =</span> <span class="si">{:7.3f}</span> <span class="s2">(best =</span> <span class="si">{:7.3f}</span><span class="s2">), noise_scale =</span> <span class="si">{}</span><span class="s2">"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span>
                <span class="n">i_episode</span><span class="p">,</span> <span class="n">agent</span><span class="o">.</span><span class="n">score</span><span class="p">,</span> <span class="n">agent</span><span class="o">.</span><span class="n">best_score</span><span class="p">,</span> <span class="n">agent</span><span class="o">.</span><span class="n">noise_scale</span><span class="p">),</span> <span class="n">end</span><span class="o">=</span><span class="s2">""</span><span class="p">)</span>  <span class="c1"># [debug]</span>
            <span class="k">break</span>
    <span class="n">sys</span><span class="o">.</span><span class="n">stdout</span><span class="o">.</span><span class="n">flush</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Episode = 1000, score =   0.607 (best =   0.617), noise_scale = 3.25</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">This agent should perform very poorly on this task. And that's where you come in!</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">## Define the Task, Design the Agent, and Train Your Agent![¶](#Define-the-Task,-Design-the-Agent,-and-Train-Your-Agent!) Amend `task.py` to specify a task of your choosing. If you're unsure what kind of task to specify, you may like to teach your quadcopter to takeoff, hover in place, land softly, or reach a target pose. After specifying your task, use the sample agent in `agents/policy_search.py` as a template to define your own agent in `agents/agent.py`. You can borrow whatever you need from the sample agent, including ideas on how you might modularize your code (using helper methods like `act()`, `learn()`, `reset_episode()`, etc.). Note that it is **highly unlikely** that the first agent and task that you specify will learn well. You will likely have to tweak various hyperparameters and the reward function for your task until you arrive at reasonably good behavior. As you develop your agent, it's important to keep an eye on how it's performing. Use the code above as inspiration to build in a mechanism to log/save the total rewards obtained in each episode to file. If the episode rewards are gradually increasing, this is an indication that your agent is learning.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [59]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">sys</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">from</span> <span class="nn">agents.agent</span> <span class="k">import</span> <span class="n">DDPG</span>
<span class="kn">from</span> <span class="nn">task</span> <span class="k">import</span> <span class="n">Task</span>

<span class="n">num_episodes</span> <span class="o">=</span> <span class="mi">1000</span>
<span class="n">init_pose</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span> <span class="mf">1.</span><span class="p">,</span><span class="mf">0.</span><span class="p">,</span><span class="mf">0.</span><span class="p">,</span><span class="mf">0.</span><span class="p">])</span>
<span class="c1">#init_velocities = np.array([0., 0., 500.])</span>
<span class="n">target_pos</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mf">0.</span><span class="p">,</span> <span class="mf">0.</span><span class="p">,</span> <span class="mf">58.</span><span class="p">])</span>
<span class="n">graph</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="mi">1001</span><span class="p">))</span>
<span class="n">task</span> <span class="o">=</span> <span class="n">Task</span><span class="p">(</span><span class="n">init_pose</span><span class="o">=</span><span class="n">init_pose</span><span class="p">,</span><span class="n">target_pos</span><span class="o">=</span><span class="n">target_pos</span><span class="p">,</span><span class="n">init_velocities</span><span class="o">=</span><span class="n">init_velocities</span><span class="p">)</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">DDPG</span><span class="p">(</span><span class="n">task</span><span class="p">)</span> 
<span class="n">reward_total</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">counter</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">best_score</span> <span class="o">=</span> <span class="o">-</span><span class="mi">5</span>

<span class="k">for</span> <span class="n">i_episode</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">num_episodes</span><span class="o">+</span><span class="mi">1</span><span class="p">):</span>
    <span class="n">state</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">reset_episode</span><span class="p">()</span> <span class="c1"># start a new episode</span>
    <span class="k">while</span> <span class="kc">True</span><span class="p">:</span>
        <span class="n">action</span> <span class="o">=</span> <span class="n">agent</span><span class="o">.</span><span class="n">act</span><span class="p">(</span><span class="n">state</span><span class="p">)</span> 
        <span class="n">next_state</span><span class="p">,</span> <span class="n">reward</span><span class="p">,</span> <span class="n">done</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>
        <span class="n">agent</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">action</span><span class="p">,</span><span class="n">reward</span><span class="p">,</span><span class="n">next_state</span><span class="p">,</span> <span class="n">done</span><span class="p">)</span>
        <span class="n">state</span> <span class="o">=</span> <span class="n">next_state</span>
        <span class="n">reward_total</span><span class="o">=</span><span class="n">reward_total</span> <span class="o">+</span> <span class="n">reward</span>
        <span class="n">counter</span> <span class="o">=</span> <span class="n">counter</span> <span class="o">+</span> <span class="mi">1</span>
        <span class="k">if</span> <span class="n">reward</span><span class="o">>=</span><span class="mf">0.98</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="s2">"Height: "</span><span class="p">)</span>
            <span class="nb">print</span><span class="p">(</span><span class="n">next_state</span><span class="p">[</span><span class="mi">2</span><span class="p">])</span>
        <span class="k">if</span> <span class="n">done</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="s2">"</span><span class="se">\r</span><span class="s2">Episode =</span> <span class="si">{:4d}</span><span class="s2">, score = (best =</span> <span class="si">{:7.3f}</span><span class="s2">), reward = "</span><span class="o">.</span><span class="n">format</span><span class="p">(</span>
                <span class="n">i_episode</span><span class="p">,</span> <span class="n">best_score</span><span class="p">),</span> <span class="n">end</span><span class="o">=</span><span class="s2">""</span><span class="p">)</span>  <span class="c1"># [debug]</span>
            <span class="n">reward_total</span> <span class="o">=</span> <span class="n">reward_total</span><span class="o">/</span><span class="n">counter</span>
            <span class="nb">print</span><span class="p">(</span><span class="n">reward_total</span><span class="p">)</span>
            <span class="k">if</span> <span class="n">reward_total</span><span class="o">></span><span class="n">best_score</span><span class="p">:</span>
                <span class="n">best_score</span> <span class="o">=</span> <span class="n">reward_total</span>
            <span class="k">if</span> <span class="n">reward_total</span><span class="o">!=-</span><span class="mi">5</span><span class="p">:</span>
                <span class="n">graph</span><span class="p">[</span><span class="n">i_episode</span><span class="p">]</span> <span class="o">=</span> <span class="n">reward_total</span>
            <span class="n">reward_total</span> <span class="o">=</span> <span class="mi">0</span>
            <span class="n">counter</span> <span class="o">=</span> <span class="mi">0</span>

            <span class="k">break</span>
    <span class="n">sys</span><span class="o">.</span><span class="n">stdout</span><span class="o">.</span><span class="n">flush</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre> <actor.actor object="" at="" 0x7f048d01c5f8="">Episode =    1, score = (best =  -5.000), reward = 0.560537343411
Episode =    2, score = (best =   0.561), reward = 0.5605428863
Episode =    3, score = (best =   0.561), reward = 0.56053623985
Episode =    4, score = (best =   0.561), reward = 0.560524015865
Episode =    5, score = (best =   0.561), reward = 0.560535341568
Episode =    6, score = (best =   0.561), reward = 0.560538567398
Episode =    7, score = (best =   0.561), reward = 0.560538540918
Episode =    8, score = (best =   0.561), reward = 0.560536147658
Episode =    9, score = (best =   0.561), reward = 0.56053403651
Episode =   10, score = (best =   0.561), reward = 0.560468835143
Episode =   11, score = (best =   0.561), reward = 0.559698091994
Episode =   12, score = (best =   0.561), reward = 0.560840430888
Episode =   13, score = (best =   0.561), reward = 0.585547117821
Height: 
56.5795819428
Height: 
57.4564307228
Height: 
58.3224396469
Episode =   14, score = (best =   0.586), reward = 0.79075525745
Height: 
56.6479302846
Height: 
57.553635622
Height: 
58.4586345456
Episode =   15, score = (best =   0.791), reward = 0.783393900293
Height: 
57.083552565
Height: 
57.7652929791
Height: 
58.4308115267
Height: 
59.0800885585
Episode =   16, score = (best =   0.791), reward = 0.79231699078
Height: 
56.5081108908
Height: 
57.490468661
Height: 
58.4644896733
Episode =   17, score = (best =   0.792), reward = 0.786005170907
Episode =   18, score = (best =   0.792), reward = 0.776381824035
Episode =   19, score = (best =   0.792), reward = 0.732275580852
Episode =   20, score = (best =   0.792), reward = 0.767474367991
Height: 
56.807719791
Height: 
57.3605201652
Height: 
57.9289078125
Height: 
58.5157149925
Episode =   21, score = (best =   0.792), reward = 0.791031350181
Height: 
56.3383004492
Height: 
57.3995244036
Height: 
58.4629242612
Episode =   22, score = (best =   0.792), reward = 0.783486045873
Height: 
57.1213533156
Height: 
58.1898665042
Episode =   23, score = (best =   0.792), reward = 0.783113140668
Episode =   24, score = (best =   0.792), reward = 0.776043939155
Height: 
56.9394682241
Height: 
57.8440231506
Height: 
58.7456998459
Episode =   25, score = (best =   0.792), reward = 0.785566580418
Height: 
57.0069378758
Height: 
57.8595224223
Height: 
58.700577551
Episode =   26, score = (best =   0.792), reward = 0.786560806275
Episode =   27, score = (best =   0.792), reward = 0.762961645966
Episode =   28, score = (best =   0.792), reward = 0.738394876831
Height: 
56.5788798656
Height: 
57.6113702423
Height: 
58.6442662263
Episode =   29, score = (best =   0.792), reward = 0.788190924386
Episode =   30, score = (best =   0.792), reward = 0.767519325704
Height: 
56.4383968793
Height: 
57.466602409
Height: 
58.4956015936
Episode =   31, score = (best =   0.792), reward = 0.784186018075
Height: 
57.2832111775
Height: 
58.3587772746
Episode =   32, score = (best =   0.792), reward = 0.786998038569
Height: 
56.6668890105
Height: 
57.652238542
Height: 
58.63688446
Episode =   33, score = (best =   0.792), reward = 0.788649376027
Height: 
56.4493648674
Height: 
57.3954394727
Height: 
58.3477173006
Episode =   34, score = (best =   0.792), reward = 0.787514234296
Height: 
57.1299646281
Height: 
58.1687818272
Episode =   35, score = (best =   0.792), reward = 0.783726308976
Episode =   36, score = (best =   0.792), reward = 0.764271072557
Height: 
56.8036617708
Height: 
57.4586294616
Height: 
58.0915964312
Height: 
58.7018893608
Episode =   37, score = (best =   0.792), reward = 0.79631152704
Episode =   38, score = (best =   0.796), reward = 0.755974330752
Episode =   39, score = (best =   0.796), reward = 0.766488273017
Height: 
56.5676904712
Height: 
57.4661902432
Height: 
58.3566153957
Episode =   40, score = (best =   0.796), reward = 0.790003064968
Height: 
56.8714544095
Height: 
57.4520842769
Height: 
58.0248762431
Height: 
58.5904737312
Episode =   41, score = (best =   0.796), reward = 0.792171138842
Height: 
57.254706864
Height: 
58.273817174
Episode =   42, score = (best =   0.796), reward = 0.783844356561
Height: 
56.4013039665
Height: 
57.2496628229
Height: 
58.0872278787
Height: 
58.9139088462
Episode =   43, score = (best =   0.796), reward = 0.791166991643
Height: 
56.7242022685
Height: 
57.6915064747
Height: 
58.6459720097
Episode =   44, score = (best =   0.796), reward = 0.790355950503
Height: 
56.3381680424
Height: 
57.3826330176
Height: 
58.4323323963
Episode =   45, score = (best =   0.796), reward = 0.783508866889
Episode =   46, score = (best =   0.796), reward = 0.771670549779
Height: 
56.5567061773
Height: 
57.590392773
Height: 
58.6239657479
Episode =   47, score = (best =   0.796), reward = 0.784273447349
Height: 
57.1430559565
Height: 
58.1241073733
Episode =   48, score = (best =   0.796), reward = 0.785472744368
Height: 
56.8916837511
Height: 
57.2439105277
Height: 
57.5643760252
Height: 
57.8524412342
Height: 
58.1075179065
Height: 
58.3290748287
Height: 
58.5166276282
Episode =   49, score = (best =   0.796), reward = 0.819493527086
Episode =   50, score = (best =   0.819), reward = 0.758093329488
Height: 
56.4352891663
Height: 
57.4248660492
Height: 
58.4051034702
Episode =   51, score = (best =   0.819), reward = 0.785997468252
Height: 
57.2785874967
Height: 
58.2272738802
Episode =   52, score = (best =   0.819), reward = 0.785379640168
Height: 
56.515094416
Height: 
57.1597997485
Height: 
57.779123472
Height: 
58.3719744131
Height: 
58.9372493556
Episode =   53, score = (best =   0.819), reward = 0.797634673731
Height: 
56.6249993864
Height: 
57.6067732441
Height: 
58.5863707422
Episode =   54, score = (best =   0.819), reward = 0.788658176728
Episode =   55, score = (best =   0.819), reward = 0.762315894904
Height: 
57.2305364488
Height: 
58.290157205
Episode =   56, score = (best =   0.819), reward = 0.78389359359
Height: 
56.3337971661
Height: 
57.3517575993
Height: 
58.360558433
Episode =   57, score = (best =   0.819), reward = 0.789415720492
Episode =   58, score = (best =   0.819), reward = 0.747672309143
Episode =   59, score = (best =   0.819), reward = 0.740269676659
Height: 
56.4791186378
Height: 
57.4170706254
Height: 
58.3652754822
Episode =   60, score = (best =   0.819), reward = 0.788042631445
Episode =   61, score = (best =   0.819), reward = 0.623977467967
Height: 
56.8855559145
Height: 
57.8995303638
Height: 
58.9176200063
Episode =   62, score = (best =   0.819), reward = 0.784429336165
Episode =   63, score = (best =   0.819), reward = 0.747687170368
Height: 
56.5995628929
Height: 
57.185632987
Height: 
57.7459505553
Height: 
58.2795166071
Height: 
58.7853574755
Episode =   64, score = (best =   0.819), reward = 0.795738631872
Height: 
56.9761073871
Height: 
57.6864815695
Height: 
58.3929454255
Height: 
59.0962883087
Episode =   65, score = (best =   0.819), reward = 0.793275976424
Episode =   66, score = (best =   0.819), reward = 0.754572622836
Episode =   67, score = (best =   0.819), reward = 0.559770829395
Episode =   68, score = (best =   0.819), reward = 0.560240764601
Episode =   69, score = (best =   0.819), reward = 0.559760995406
Episode =   70, score = (best =   0.819), reward = 0.559783763259
Episode =   71, score = (best =   0.819), reward = 0.560234972281
Episode =   72, score = (best =   0.819), reward = 0.559802451092
Episode =   73, score = (best =   0.819), reward = 0.560264056133
Episode =   74, score = (best =   0.819), reward = 0.560240633652
Episode =   75, score = (best =   0.819), reward = 0.560273943076
Episode =   76, score = (best =   0.819), reward = 0.560269735352
Episode =   77, score = (best =   0.819), reward = 0.560217424033
Episode =   78, score = (best =   0.819), reward = 0.559794043258
Episode =   79, score = (best =   0.819), reward = 0.559751619982
Episode =   80, score = (best =   0.819), reward = 0.560231027527
Episode =   81, score = (best =   0.819), reward = 0.560212622724
Episode =   82, score = (best =   0.819), reward = 0.55976200674
Episode =   83, score = (best =   0.819), reward = 0.560257851788
Episode =   84, score = (best =   0.819), reward = 0.560251258274
Episode =   85, score = (best =   0.819), reward = 0.560253598689
Episode =   86, score = (best =   0.819), reward = 0.560259576873
Episode =   87, score = (best =   0.819), reward = 0.560237151271
Episode =   88, score = (best =   0.819), reward = 0.560202542213
Episode =   89, score = (best =   0.819), reward = 0.560251595944
Episode =   90, score = (best =   0.819), reward = 0.56022363327
Episode =   91, score = (best =   0.819), reward = 0.55977712009
Episode =   92, score = (best =   0.819), reward = 0.559813383016
Episode =   93, score = (best =   0.819), reward = 0.560199663593
Episode =   94, score = (best =   0.819), reward = 0.560263777776
Episode =   95, score = (best =   0.819), reward = 0.56024565968
Episode =   96, score = (best =   0.819), reward = 0.559806773139
Episode =   97, score = (best =   0.819), reward = 0.560221642741
Episode =   98, score = (best =   0.819), reward = 0.560239789531
Episode =   99, score = (best =   0.819), reward = 0.560242570977
Episode =  100, score = (best =   0.819), reward = 0.560249188821
Episode =  101, score = (best =   0.819), reward = 0.560228778037
Episode =  102, score = (best =   0.819), reward = 0.559760490675
Episode =  103, score = (best =   0.819), reward = 0.560218204674
Episode =  104, score = (best =   0.819), reward = 0.56016873656
Episode =  105, score = (best =   0.819), reward = 0.560235483424
Episode =  106, score = (best =   0.819), reward = 0.560243915197
Episode =  107, score = (best =   0.819), reward = 0.559773671905
Episode =  108, score = (best =   0.819), reward = 0.559835292996
Episode =  109, score = (best =   0.819), reward = 0.559760588648
Episode =  110, score = (best =   0.819), reward = 0.559872723701
Episode =  111, score = (best =   0.819), reward = 0.560275700622
Episode =  112, score = (best =   0.819), reward = 0.559808935988
Episode =  113, score = (best =   0.819), reward = 0.55978912669
Episode =  114, score = (best =   0.819), reward = 0.559777442905
Episode =  115, score = (best =   0.819), reward = 0.559761704401
Episode =  116, score = (best =   0.819), reward = 0.559789014697
Episode =  117, score = (best =   0.819), reward = 0.559767897832
Episode =  118, score = (best =   0.819), reward = 0.55977341935
Episode =  119, score = (best =   0.819), reward = 0.560265520553
Episode =  120, score = (best =   0.819), reward = 0.559791716514
Episode =  121, score = (best =   0.819), reward = 0.559755520522
Episode =  122, score = (best =   0.819), reward = 0.560237564689
Episode =  123, score = (best =   0.819), reward = 0.560263480242
Episode =  124, score = (best =   0.819), reward = 0.559782391071
Episode =  125, score = (best =   0.819), reward = 0.560225588936
Episode =  126, score = (best =   0.819), reward = 0.560234342836
Episode =  127, score = (best =   0.819), reward = 0.560203272903
Episode =  128, score = (best =   0.819), reward = 0.559752560242
Episode =  129, score = (best =   0.819), reward = 0.560249395106
Episode =  130, score = (best =   0.819), reward = 0.560218462242
Episode =  131, score = (best =   0.819), reward = 0.559756286553
Episode =  132, score = (best =   0.819), reward = 0.559797374905
Episode =  133, score = (best =   0.819), reward = 0.560256373069
Episode =  134, score = (best =   0.819), reward = 0.559745048036
Episode =  135, score = (best =   0.819), reward = 0.560247089437
Episode =  136, score = (best =   0.819), reward = 0.560260572522
Episode =  137, score = (best =   0.819), reward = 0.560218178826
Episode =  138, score = (best =   0.819), reward = 0.559802624398
Episode =  139, score = (best =   0.819), reward = 0.559753095048
Episode =  140, score = (best =   0.819), reward = 0.560260641175
Episode =  141, score = (best =   0.819), reward = 0.560268476798
Episode =  142, score = (best =   0.819), reward = 0.560219144403
Episode =  143, score = (best =   0.819), reward = 0.560213143903
Episode =  144, score = (best =   0.819), reward = 0.559761097059
Episode =  145, score = (best =   0.819), reward = 0.560245854423
Episode =  146, score = (best =   0.819), reward = 0.559815642733
Episode =  147, score = (best =   0.819), reward = 0.560245242981
Episode =  148, score = (best =   0.819), reward = 0.560231233542
Episode =  149, score = (best =   0.819), reward = 0.559760670674
Episode =  150, score = (best =   0.819), reward = 0.559805055398
Episode =  151, score = (best =   0.819), reward = 0.560273119755
Episode =  152, score = (best =   0.819), reward = 0.559758094729
Episode =  153, score = (best =   0.819), reward = 0.5597523557
Episode =  154, score = (best =   0.819), reward = 0.559794976903
Episode =  155, score = (best =   0.819), reward = 0.559750586636
Episode =  156, score = (best =   0.819), reward = 0.560215135979
Episode =  157, score = (best =   0.819), reward = 0.559750150557
Episode =  158, score = (best =   0.819), reward = 0.560245181408
Episode =  159, score = (best =   0.819), reward = 0.560257242956
Episode =  160, score = (best =   0.819), reward = 0.56018113786
Episode =  161, score = (best =   0.819), reward = 0.559766665018
Episode =  162, score = (best =   0.819), reward = 0.559809636513
Episode =  163, score = (best =   0.819), reward = 0.560262890561
Episode =  164, score = (best =   0.819), reward = 0.560186317569
Episode =  165, score = (best =   0.819), reward = 0.55975574507
Episode =  166, score = (best =   0.819), reward = 0.55980741417
Episode =  167, score = (best =   0.819), reward = 0.560199961071
Episode =  168, score = (best =   0.819), reward = 0.559762633977
Episode =  169, score = (best =   0.819), reward = 0.559775785938
Episode =  170, score = (best =   0.819), reward = 0.559789544437
Episode =  171, score = (best =   0.819), reward = 0.559759453051
Episode =  172, score = (best =   0.819), reward = 0.560273233761
Episode =  173, score = (best =   0.819), reward = 0.560236988356
Episode =  174, score = (best =   0.819), reward = 0.559762708268
Episode =  175, score = (best =   0.819), reward = 0.559790318754
Episode =  176, score = (best =   0.819), reward = 0.559773439228
Episode =  177, score = (best =   0.819), reward = 0.559794187764
Episode =  178, score = (best =   0.819), reward = 0.559756900405
Episode =  179, score = (best =   0.819), reward = 0.560257988802
Episode =  180, score = (best =   0.819), reward = 0.560221198992
Episode =  181, score = (best =   0.819), reward = 0.560240969669
Episode =  182, score = (best =   0.819), reward = 0.559798774942
Episode =  183, score = (best =   0.819), reward = 0.559779640641
Episode =  184, score = (best =   0.819), reward = 0.559795592853
Episode =  185, score = (best =   0.819), reward = 0.560187398978
Episode =  186, score = (best =   0.819), reward = 0.560263176771
Episode =  187, score = (best =   0.819), reward = 0.560250107126
Episode =  188, score = (best =   0.819), reward = 0.559763303914
Episode =  189, score = (best =   0.819), reward = 0.559759430955
Episode =  190, score = (best =   0.819), reward = 0.559765109469
Episode =  191, score = (best =   0.819), reward = 0.560252236946
Episode =  192, score = (best =   0.819), reward = 0.560244386914
Episode =  193, score = (best =   0.819), reward = 0.559773856097
Episode =  194, score = (best =   0.819), reward = 0.560253305263
Episode =  195, score = (best =   0.819), reward = 0.559860151826
Episode =  196, score = (best =   0.819), reward = 0.560261146797
Episode =  197, score = (best =   0.819), reward = 0.560231448215
Episode =  198, score = (best =   0.819), reward = 0.56020520116
Episode =  199, score = (best =   0.819), reward = 0.560227822359
Episode =  200, score = (best =   0.819), reward = 0.560174883759
Episode =  201, score = (best =   0.819), reward = 0.559761036188
Episode =  202, score = (best =   0.819), reward = 0.56022436799
Episode =  203, score = (best =   0.819), reward = 0.559788004596
Episode =  204, score = (best =   0.819), reward = 0.559787434657
Episode =  205, score = (best =   0.819), reward = 0.559770989849
Episode =  206, score = (best =   0.819), reward = 0.55979259157
Episode =  207, score = (best =   0.819), reward = 0.560110351322
Episode =  208, score = (best =   0.819), reward = 0.560216710525
Episode =  209, score = (best =   0.819), reward = 0.559750017854
Episode =  210, score = (best =   0.819), reward = 0.559753074008
Episode =  211, score = (best =   0.819), reward = 0.560209924809
Episode =  212, score = (best =   0.819), reward = 0.560197412535
Episode =  213, score = (best =   0.819), reward = 0.560259686486
Episode =  214, score = (best =   0.819), reward = 0.560257010044
Episode =  215, score = (best =   0.819), reward = 0.559791317614
Episode =  216, score = (best =   0.819), reward = 0.560253256434
Episode =  217, score = (best =   0.819), reward = 0.560240468122
Episode =  218, score = (best =   0.819), reward = 0.560245926786
Episode =  219, score = (best =   0.819), reward = 0.560210523438
Episode =  220, score = (best =   0.819), reward = 0.560268310083
Episode =  221, score = (best =   0.819), reward = 0.559765454133
Episode =  222, score = (best =   0.819), reward = 0.560257693998
Episode =  223, score = (best =   0.819), reward = 0.559749267581
Episode =  224, score = (best =   0.819), reward = 0.559790257166
Episode =  225, score = (best =   0.819), reward = 0.560251842387
Episode =  226, score = (best =   0.819), reward = 0.559819341592
Episode =  227, score = (best =   0.819), reward = 0.559760868847
Episode =  228, score = (best =   0.819), reward = 0.560266751272
Episode =  229, score = (best =   0.819), reward = 0.55984860252
Episode =  230, score = (best =   0.819), reward = 0.560256482612
Episode =  231, score = (best =   0.819), reward = 0.560264598882
Episode =  232, score = (best =   0.819), reward = 0.560229766931
Episode =  233, score = (best =   0.819), reward = 0.56024415737
Episode =  234, score = (best =   0.819), reward = 0.559766908598
Episode =  235, score = (best =   0.819), reward = 0.559763534301
Episode =  236, score = (best =   0.819), reward = 0.559786611272
Episode =  237, score = (best =   0.819), reward = 0.559809938
Episode =  238, score = (best =   0.819), reward = 0.560251729081
Episode =  239, score = (best =   0.819), reward = 0.560220109093
Episode =  240, score = (best =   0.819), reward = 0.560189129957
Episode =  241, score = (best =   0.819), reward = 0.560246133574
Episode =  242, score = (best =   0.819), reward = 0.559777354932
Episode =  243, score = (best =   0.819), reward = 0.559772998244
Episode =  244, score = (best =   0.819), reward = 0.560214942747
Episode =  245, score = (best =   0.819), reward = 0.559752861952
Episode =  246, score = (best =   0.819), reward = 0.559778003441
Episode =  247, score = (best =   0.819), reward = 0.559774831855
Episode =  248, score = (best =   0.819), reward = 0.559842275881
Episode =  249, score = (best =   0.819), reward = 0.559851784586
Episode =  250, score = (best =   0.819), reward = 0.560268806443
Episode =  251, score = (best =   0.819), reward = 0.560209011722
Episode =  252, score = (best =   0.819), reward = 0.560236649643
Episode =  253, score = (best =   0.819), reward = 0.559813471989
Episode =  254, score = (best =   0.819), reward = 0.559796686661
Episode =  255, score = (best =   0.819), reward = 0.559827603704
Episode =  256, score = (best =   0.819), reward = 0.560250157037
Episode =  257, score = (best =   0.819), reward = 0.559772089554
Episode =  258, score = (best =   0.819), reward = 0.559787691258
Episode =  259, score = (best =   0.819), reward = 0.560209989036
Episode =  260, score = (best =   0.819), reward = 0.559780549461
Episode =  261, score = (best =   0.819), reward = 0.560251001062
Episode =  262, score = (best =   0.819), reward = 0.560226967059
Episode =  263, score = (best =   0.819), reward = 0.560271208851
Episode =  264, score = (best =   0.819), reward = 0.560270070486
Episode =  265, score = (best =   0.819), reward = 0.560272016614
Episode =  266, score = (best =   0.819), reward = 0.559769163979
Episode =  267, score = (best =   0.819), reward = 0.560197103127
Episode =  268, score = (best =   0.819), reward = 0.560189041545
Episode =  269, score = (best =   0.819), reward = 0.560255196047
Episode =  270, score = (best =   0.819), reward = 0.559793313302
Episode =  271, score = (best =   0.819), reward = 0.559761824222
Episode =  272, score = (best =   0.819), reward = 0.560234547253
Episode =  273, score = (best =   0.819), reward = 0.559864023094
Episode =  274, score = (best =   0.819), reward = 0.560216065742
Episode =  275, score = (best =   0.819), reward = 0.559762392693
Episode =  276, score = (best =   0.819), reward = 0.560257120415
Episode =  277, score = (best =   0.819), reward = 0.560222510572
Episode =  278, score = (best =   0.819), reward = 0.56020004203
Episode =  279, score = (best =   0.819), reward = 0.560273667901
Episode =  280, score = (best =   0.819), reward = 0.559863191167
Episode =  281, score = (best =   0.819), reward = 0.560273511777
Episode =  282, score = (best =   0.819), reward = 0.560256334589
Episode =  283, score = (best =   0.819), reward = 0.560269592284
Episode =  284, score = (best =   0.819), reward = 0.560248934899
Episode =  285, score = (best =   0.819), reward = 0.559766982785
Episode =  286, score = (best =   0.819), reward = 0.560261950468
Episode =  287, score = (best =   0.819), reward = 0.559783693747
Episode =  288, score = (best =   0.819), reward = 0.560257060019
Episode =  289, score = (best =   0.819), reward = 0.560194339628
Episode =  290, score = (best =   0.819), reward = 0.560210493396
Episode =  291, score = (best =   0.819), reward = 0.559812158305
Episode =  292, score = (best =   0.819), reward = 0.560271485527
Episode =  293, score = (best =   0.819), reward = 0.56025708522
Episode =  294, score = (best =   0.819), reward = 0.559776411371
Episode =  295, score = (best =   0.819), reward = 0.559754000322
Episode =  296, score = (best =   0.819), reward = 0.560270439472
Episode =  297, score = (best =   0.819), reward = 0.559748400643
Episode =  298, score = (best =   0.819), reward = 0.559802387776
Episode =  299, score = (best =   0.819), reward = 0.560271837806
Episode =  300, score = (best =   0.819), reward = 0.560264294692
Episode =  301, score = (best =   0.819), reward = 0.560230276855
Episode =  302, score = (best =   0.819), reward = 0.560273282146
Episode =  303, score = (best =   0.819), reward = 0.560248096402
Episode =  304, score = (best =   0.819), reward = 0.560228976039
Episode =  305, score = (best =   0.819), reward = 0.56024822062
Episode =  306, score = (best =   0.819), reward = 0.559808411758
Episode =  307, score = (best =   0.819), reward = 0.560242836062
Episode =  308, score = (best =   0.819), reward = 0.55982281344
Episode =  309, score = (best =   0.819), reward = 0.560273573088
Episode =  310, score = (best =   0.819), reward = 0.559798771791
Episode =  311, score = (best =   0.819), reward = 0.560259921219
Episode =  312, score = (best =   0.819), reward = 0.560233434931
Episode =  313, score = (best =   0.819), reward = 0.559774203973
Episode =  314, score = (best =   0.819), reward = 0.559780981705
Episode =  315, score = (best =   0.819), reward = 0.559752781953
Episode =  316, score = (best =   0.819), reward = 0.560259153475
Episode =  317, score = (best =   0.819), reward = 0.559752999584
Episode =  318, score = (best =   0.819), reward = 0.560241710805
Episode =  319, score = (best =   0.819), reward = 0.559796266515
Episode =  320, score = (best =   0.819), reward = 0.560246412311
Episode =  321, score = (best =   0.819), reward = 0.559757131704
Episode =  322, score = (best =   0.819), reward = 0.559746807918
Episode =  323, score = (best =   0.819), reward = 0.559782533108
Episode =  324, score = (best =   0.819), reward = 0.559757905022
Episode =  325, score = (best =   0.819), reward = 0.559782852232
Episode =  326, score = (best =   0.819), reward = 0.560223112791
Episode =  327, score = (best =   0.819), reward = 0.559799824451
Episode =  328, score = (best =   0.819), reward = 0.560197268759
Episode =  329, score = (best =   0.819), reward = 0.560271338225
Episode =  330, score = (best =   0.819), reward = 0.560225400141
Episode =  331, score = (best =   0.819), reward = 0.559767882519
Episode =  332, score = (best =   0.819), reward = 0.559830105567
Episode =  333, score = (best =   0.819), reward = 0.560204998329
Episode =  334, score = (best =   0.819), reward = 0.560235080311
Episode =  335, score = (best =   0.819), reward = 0.559763253429
Episode =  336, score = (best =   0.819), reward = 0.559855563947
Episode =  337, score = (best =   0.819), reward = 0.560205488073
Episode =  338, score = (best =   0.819), reward = 0.560213704698
Episode =  339, score = (best =   0.819), reward = 0.560240495979
Episode =  340, score = (best =   0.819), reward = 0.560252050669
Episode =  341, score = (best =   0.819), reward = 0.559772951144
Episode =  342, score = (best =   0.819), reward = 0.56023913256
Episode =  343, score = (best =   0.819), reward = 0.560254381437
Episode =  344, score = (best =   0.819), reward = 0.559806366749
Episode =  345, score = (best =   0.819), reward = 0.559851500113
Episode =  346, score = (best =   0.819), reward = 0.560239043004
Episode =  347, score = (best =   0.819), reward = 0.559832098361
Episode =  348, score = (best =   0.819), reward = 0.560229428121
Episode =  349, score = (best =   0.819), reward = 0.560266561333
Episode =  350, score = (best =   0.819), reward = 0.559750258993
Episode =  351, score = (best =   0.819), reward = 0.559752325041
Episode =  352, score = (best =   0.819), reward = 0.559776462912
Episode =  353, score = (best =   0.819), reward = 0.560266931325
Episode =  354, score = (best =   0.819), reward = 0.559786532849
Episode =  355, score = (best =   0.819), reward = 0.559750247626
Episode =  356, score = (best =   0.819), reward = 0.559749858052
Episode =  357, score = (best =   0.819), reward = 0.560215359832
Episode =  358, score = (best =   0.819), reward = 0.559762612101
Episode =  359, score = (best =   0.819), reward = 0.560243074433
Episode =  360, score = (best =   0.819), reward = 0.560220328753
Episode =  361, score = (best =   0.819), reward = 0.560268446131
Episode =  362, score = (best =   0.819), reward = 0.559774403172
Episode =  363, score = (best =   0.819), reward = 0.559785761124
Episode =  364, score = (best =   0.819), reward = 0.560240676618
Episode =  365, score = (best =   0.819), reward = 0.55980399327
Episode =  366, score = (best =   0.819), reward = 0.560231071198
Episode =  367, score = (best =   0.819), reward = 0.560165979247
Episode =  368, score = (best =   0.819), reward = 0.560158262263
Episode =  369, score = (best =   0.819), reward = 0.560192687229
Episode =  370, score = (best =   0.819), reward = 0.559751535006
Episode =  371, score = (best =   0.819), reward = 0.559762034602
Episode =  372, score = (best =   0.819), reward = 0.56026285796
Episode =  373, score = (best =   0.819), reward = 0.559806932724
Episode =  374, score = (best =   0.819), reward = 0.560220562257
Episode =  375, score = (best =   0.819), reward = 0.56024016179
Episode =  376, score = (best =   0.819), reward = 0.560162886423
Episode =  377, score = (best =   0.819), reward = 0.559817744047
Episode =  378, score = (best =   0.819), reward = 0.560240119177
Episode =  379, score = (best =   0.819), reward = 0.560272298546
Episode =  380, score = (best =   0.819), reward = 0.560178534049
Episode =  381, score = (best =   0.819), reward = 0.560197032024
Episode =  382, score = (best =   0.819), reward = 0.559793020186
Episode =  383, score = (best =   0.819), reward = 0.559786426156
Episode =  384, score = (best =   0.819), reward = 0.559789449147
Episode =  385, score = (best =   0.819), reward = 0.56026060406
Episode =  386, score = (best =   0.819), reward = 0.559750964083
Episode =  387, score = (best =   0.819), reward = 0.559791864353
Episode =  388, score = (best =   0.819), reward = 0.55977850557
Episode =  389, score = (best =   0.819), reward = 0.560229986871
Episode =  390, score = (best =   0.819), reward = 0.560241767726
Episode =  391, score = (best =   0.819), reward = 0.559780802414
Episode =  392, score = (best =   0.819), reward = 0.559771340922
Episode =  393, score = (best =   0.819), reward = 0.560210575214
Episode =  394, score = (best =   0.819), reward = 0.559776363696
Episode =  395, score = (best =   0.819), reward = 0.560218140906
Episode =  396, score = (best =   0.819), reward = 0.559776764464
Episode =  397, score = (best =   0.819), reward = 0.559868986934
Episode =  398, score = (best =   0.819), reward = 0.560255251842
Episode =  399, score = (best =   0.819), reward = 0.559785657943
Episode =  400, score = (best =   0.819), reward = 0.560211516035
Episode =  401, score = (best =   0.819), reward = 0.559760419576
Episode =  402, score = (best =   0.819), reward = 0.559766867248
Episode =  403, score = (best =   0.819), reward = 0.560221766927
Episode =  404, score = (best =   0.819), reward = 0.560249471897
Episode =  405, score = (best =   0.819), reward = 0.560234251906
Episode =  406, score = (best =   0.819), reward = 0.560263032355
Episode =  407, score = (best =   0.819), reward = 0.560230882264
Episode =  408, score = (best =   0.819), reward = 0.560223083208
Episode =  409, score = (best =   0.819), reward = 0.560241689458
Episode =  410, score = (best =   0.819), reward = 0.560238588918
Episode =  411, score = (best =   0.819), reward = 0.559782887649
Episode =  412, score = (best =   0.819), reward = 0.560267605576
Episode =  413, score = (best =   0.819), reward = 0.559779830761
Episode =  414, score = (best =   0.819), reward = 0.559760835775
Episode =  415, score = (best =   0.819), reward = 0.55975764468
Episode =  416, score = (best =   0.819), reward = 0.560238463397
Episode =  417, score = (best =   0.819), reward = 0.559749340728
Episode =  418, score = (best =   0.819), reward = 0.559752349299
Episode =  419, score = (best =   0.819), reward = 0.559788849239
Episode =  420, score = (best =   0.819), reward = 0.56025344965
Episode =  421, score = (best =   0.819), reward = 0.559754188298
Episode =  422, score = (best =   0.819), reward = 0.559776781536
Episode =  423, score = (best =   0.819), reward = 0.560216086086
Episode =  424, score = (best =   0.819), reward = 0.56019554246
Episode =  425, score = (best =   0.819), reward = 0.559811348261
Episode =  426, score = (best =   0.819), reward = 0.56024207443
Episode =  427, score = (best =   0.819), reward = 0.559805074741
Episode =  428, score = (best =   0.819), reward = 0.559814668949
Episode =  429, score = (best =   0.819), reward = 0.560245491047
Episode =  430, score = (best =   0.819), reward = 0.560263179743
Episode =  431, score = (best =   0.819), reward = 0.560221850664
Episode =  432, score = (best =   0.819), reward = 0.56021440326
Episode =  433, score = (best =   0.819), reward = 0.560254954359
Episode =  434, score = (best =   0.819), reward = 0.559788719672
Episode =  435, score = (best =   0.819), reward = 0.560246440243
Episode =  436, score = (best =   0.819), reward = 0.559779924754
Episode =  437, score = (best =   0.819), reward = 0.560224327991
Episode =  438, score = (best =   0.819), reward = 0.559814429417
Episode =  439, score = (best =   0.819), reward = 0.560205516853
Episode =  440, score = (best =   0.819), reward = 0.559776262709
Episode =  441, score = (best =   0.819), reward = 0.560215174174
Episode =  442, score = (best =   0.819), reward = 0.560251956522
Episode =  443, score = (best =   0.819), reward = 0.559777012963
Episode =  444, score = (best =   0.819), reward = 0.559805386537
Episode =  445, score = (best =   0.819), reward = 0.560271636014
Episode =  446, score = (best =   0.819), reward = 0.560223142125
Episode =  447, score = (best =   0.819), reward = 0.559762297298
Episode =  448, score = (best =   0.819), reward = 0.560234878421
Episode =  449, score = (best =   0.819), reward = 0.559762123322
Episode =  450, score = (best =   0.819), reward = 0.559754723487
Episode =  451, score = (best =   0.819), reward = 0.560211144728
Episode =  452, score = (best =   0.819), reward = 0.560239111878
Episode =  453, score = (best =   0.819), reward = 0.56015536932
Episode =  454, score = (best =   0.819), reward = 0.560249913751
Episode =  455, score = (best =   0.819), reward = 0.560238414748
Episode =  456, score = (best =   0.819), reward = 0.560265913233
Episode =  457, score = (best =   0.819), reward = 0.560254740606
Episode =  458, score = (best =   0.819), reward = 0.559793164805
Episode =  459, score = (best =   0.819), reward = 0.560271856465
Episode =  460, score = (best =   0.819), reward = 0.559797345912
Episode =  461, score = (best =   0.819), reward = 0.559772265942
Episode =  462, score = (best =   0.819), reward = 0.560204595592
Episode =  463, score = (best =   0.819), reward = 0.560220204286
Episode =  464, score = (best =   0.819), reward = 0.560216665896
Episode =  465, score = (best =   0.819), reward = 0.560272396723
Episode =  466, score = (best =   0.819), reward = 0.560268660089
Episode =  467, score = (best =   0.819), reward = 0.56023469198
Episode =  468, score = (best =   0.819), reward = 0.559836859507
Episode =  469, score = (best =   0.819), reward = 0.559783517421
Episode =  470, score = (best =   0.819), reward = 0.55981953576
Episode =  471, score = (best =   0.819), reward = 0.560240866069
Episode =  472, score = (best =   0.819), reward = 0.559839847994
Episode =  473, score = (best =   0.819), reward = 0.559768915507
Episode =  474, score = (best =   0.819), reward = 0.560258425627
Episode =  475, score = (best =   0.819), reward = 0.55978974269
Episode =  476, score = (best =   0.819), reward = 0.559779588241
Episode =  477, score = (best =   0.819), reward = 0.559791124327
Episode =  478, score = (best =   0.819), reward = 0.559774886529
Episode =  479, score = (best =   0.819), reward = 0.55975360969
Episode =  480, score = (best =   0.819), reward = 0.559776773076
Episode =  481, score = (best =   0.819), reward = 0.560272596251
Episode =  482, score = (best =   0.819), reward = 0.560218448053
Episode =  483, score = (best =   0.819), reward = 0.559790920051
Episode =  484, score = (best =   0.819), reward = 0.559749141849
Episode =  485, score = (best =   0.819), reward = 0.559747507363
Episode =  486, score = (best =   0.819), reward = 0.560244048373
Episode =  487, score = (best =   0.819), reward = 0.560242722175
Episode =  488, score = (best =   0.819), reward = 0.560268818463
Episode =  489, score = (best =   0.819), reward = 0.560223498097
Episode =  490, score = (best =   0.819), reward = 0.56025845683
Episode =  491, score = (best =   0.819), reward = 0.560232045963
Episode =  492, score = (best =   0.819), reward = 0.55977653894
Episode =  493, score = (best =   0.819), reward = 0.559786576231
Episode =  494, score = (best =   0.819), reward = 0.559793761442
Episode =  495, score = (best =   0.819), reward = 0.559797798271
Episode =  496, score = (best =   0.819), reward = 0.559779597974
Episode =  497, score = (best =   0.819), reward = 0.559789824246
Episode =  498, score = (best =   0.819), reward = 0.559761983593
Episode =  499, score = (best =   0.819), reward = 0.559783621027
Episode =  500, score = (best =   0.819), reward = 0.560222989467
Episode =  501, score = (best =   0.819), reward = 0.560227951021
Episode =  502, score = (best =   0.819), reward = 0.559790081248
Episode =  503, score = (best =   0.819), reward = 0.559816456425
Episode =  504, score = (best =   0.819), reward = 0.559803662125
Episode =  505, score = (best =   0.819), reward = 0.559761883615
Episode =  506, score = (best =   0.819), reward = 0.559755692391
Episode =  507, score = (best =   0.819), reward = 0.559750135838
Episode =  508, score = (best =   0.819), reward = 0.560266769775
Episode =  509, score = (best =   0.819), reward = 0.560243970609
Episode =  510, score = (best =   0.819), reward = 0.559875544948
Episode =  511, score = (best =   0.819), reward = 0.559795711516
Episode =  512, score = (best =   0.819), reward = 0.560240904596
Episode =  513, score = (best =   0.819), reward = 0.560242180335
Episode =  514, score = (best =   0.819), reward = 0.560253280852
Episode =  515, score = (best =   0.819), reward = 0.560254547846
Episode =  516, score = (best =   0.819), reward = 0.559767842298
Episode =  517, score = (best =   0.819), reward = 0.559758295005
Episode =  518, score = (best =   0.819), reward = 0.559766829376
Episode =  519, score = (best =   0.819), reward = 0.560257003589
Episode =  520, score = (best =   0.819), reward = 0.560264865423
Episode =  521, score = (best =   0.819), reward = 0.560251188328
Episode =  522, score = (best =   0.819), reward = 0.560237874835
Episode =  523, score = (best =   0.819), reward = 0.5602601943
Episode =  524, score = (best =   0.819), reward = 0.55975217684
Episode =  525, score = (best =   0.819), reward = 0.560226672986
Episode =  526, score = (best =   0.819), reward = 0.560243625817
Episode =  527, score = (best =   0.819), reward = 0.560251979994
Episode =  528, score = (best =   0.819), reward = 0.559748346472
Episode =  529, score = (best =   0.819), reward = 0.560168095603
Episode =  530, score = (best =   0.819), reward = 0.559803932881
Episode =  531, score = (best =   0.819), reward = 0.559755634199
Episode =  532, score = (best =   0.819), reward = 0.560253098885
Episode =  533, score = (best =   0.819), reward = 0.559755188468
Episode =  534, score = (best =   0.819), reward = 0.559789222526
Episode =  535, score = (best =   0.819), reward = 0.560253764394
Episode =  536, score = (best =   0.819), reward = 0.560248219903
Episode =  537, score = (best =   0.819), reward = 0.560205542429
Episode =  538, score = (best =   0.819), reward = 0.55979192726
Episode =  539, score = (best =   0.819), reward = 0.559779735526
Episode =  540, score = (best =   0.819), reward = 0.560218367372
Episode =  541, score = (best =   0.819), reward = 0.560268790006
Episode =  542, score = (best =   0.819), reward = 0.560266152763
Episode =  543, score = (best =   0.819), reward = 0.559763050918
Episode =  544, score = (best =   0.819), reward = 0.560255023365
Episode =  545, score = (best =   0.819), reward = 0.560205636015
Episode =  546, score = (best =   0.819), reward = 0.559776137096
Episode =  547, score = (best =   0.819), reward = 0.560204926955
Episode =  548, score = (best =   0.819), reward = 0.559756032841
Episode =  549, score = (best =   0.819), reward = 0.560219528863
Episode =  550, score = (best =   0.819), reward = 0.559801930387
Episode =  551, score = (best =   0.819), reward = 0.560176536563
Episode =  552, score = (best =   0.819), reward = 0.560274738555
Episode =  553, score = (best =   0.819), reward = 0.560213578717
Episode =  554, score = (best =   0.819), reward = 0.559803378685
Episode =  555, score = (best =   0.819), reward = 0.559756681634
Episode =  556, score = (best =   0.819), reward = 0.559765320204
Episode =  557, score = (best =   0.819), reward = 0.560246289047
Episode =  558, score = (best =   0.819), reward = 0.559756280009
Episode =  559, score = (best =   0.819), reward = 0.560246123477
Episode =  560, score = (best =   0.819), reward = 0.559750378263
Episode =  561, score = (best =   0.819), reward = 0.560269612523
Episode =  562, score = (best =   0.819), reward = 0.559792578212
Episode =  563, score = (best =   0.819), reward = 0.56026446878
Episode =  564, score = (best =   0.819), reward = 0.56025611526
Episode =  565, score = (best =   0.819), reward = 0.560266762794
Episode =  566, score = (best =   0.819), reward = 0.560130687194
Episode =  567, score = (best =   0.819), reward = 0.560224049137
Episode =  568, score = (best =   0.819), reward = 0.56020554745
Episode =  569, score = (best =   0.819), reward = 0.560231192011
Episode =  570, score = (best =   0.819), reward = 0.559827971924
Episode =  571, score = (best =   0.819), reward = 0.560255959499
Episode =  572, score = (best =   0.819), reward = 0.559770664427
Episode =  573, score = (best =   0.819), reward = 0.56022388638
Episode =  574, score = (best =   0.819), reward = 0.559753465842
Episode =  575, score = (best =   0.819), reward = 0.560247882395
Episode =  576, score = (best =   0.819), reward = 0.560234026736
Episode =  577, score = (best =   0.819), reward = 0.559757372679
Episode =  578, score = (best =   0.819), reward = 0.56024876155
Episode =  579, score = (best =   0.819), reward = 0.560227314682
Episode =  580, score = (best =   0.819), reward = 0.560240989975
Episode =  581, score = (best =   0.819), reward = 0.559788843292
Episode =  582, score = (best =   0.819), reward = 0.560271214977
Episode =  583, score = (best =   0.819), reward = 0.560220655431
Episode =  584, score = (best =   0.819), reward = 0.55979123157
Episode =  585, score = (best =   0.819), reward = 0.559788248467
Episode =  586, score = (best =   0.819), reward = 0.55976512454
Episode =  587, score = (best =   0.819), reward = 0.560212661455
Episode =  588, score = (best =   0.819), reward = 0.559797742628
Episode =  589, score = (best =   0.819), reward = 0.559748480662
Episode =  590, score = (best =   0.819), reward = 0.56021723043
Episode =  591, score = (best =   0.819), reward = 0.560167186371
Episode =  592, score = (best =   0.819), reward = 0.559750432634
Episode =  593, score = (best =   0.819), reward = 0.55974962193
Episode =  594, score = (best =   0.819), reward = 0.559840245145
Episode =  595, score = (best =   0.819), reward = 0.560261112077
Episode =  596, score = (best =   0.819), reward = 0.560246786128
Episode =  597, score = (best =   0.819), reward = 0.559763537622
Episode =  598, score = (best =   0.819), reward = 0.56020759269
Episode =  599, score = (best =   0.819), reward = 0.559747643357
Episode =  600, score = (best =   0.819), reward = 0.559752435477
Episode =  601, score = (best =   0.819), reward = 0.560191906175
Episode =  602, score = (best =   0.819), reward = 0.559760935825
Episode =  603, score = (best =   0.819), reward = 0.560259942461
Episode =  604, score = (best =   0.819), reward = 0.560212366386
Episode =  605, score = (best =   0.819), reward = 0.560269263517
Episode =  606, score = (best =   0.819), reward = 0.559768855467
Episode =  607, score = (best =   0.819), reward = 0.559780859797
Episode =  608, score = (best =   0.819), reward = 0.559762257389
Episode =  609, score = (best =   0.819), reward = 0.559798370657
Episode =  610, score = (best =   0.819), reward = 0.560216407289
Episode =  611, score = (best =   0.819), reward = 0.560256778297
Episode =  612, score = (best =   0.819), reward = 0.560222416493
Episode =  613, score = (best =   0.819), reward = 0.560266610591
Episode =  614, score = (best =   0.819), reward = 0.559790459291
Episode =  615, score = (best =   0.819), reward = 0.559767335699
Episode =  616, score = (best =   0.819), reward = 0.559796490182
Episode =  617, score = (best =   0.819), reward = 0.559773534443
Episode =  618, score = (best =   0.819), reward = 0.559788952011
Episode =  619, score = (best =   0.819), reward = 0.559786768668
Episode =  620, score = (best =   0.819), reward = 0.559752596615
Episode =  621, score = (best =   0.819), reward = 0.559755094173
Episode =  622, score = (best =   0.819), reward = 0.55975763797
Episode =  623, score = (best =   0.819), reward = 0.559770705616
Episode =  624, score = (best =   0.819), reward = 0.560192512445
Episode =  625, score = (best =   0.819), reward = 0.560239355612
Episode =  626, score = (best =   0.819), reward = 0.560252083601
Episode =  627, score = (best =   0.819), reward = 0.560270361133
Episode =  628, score = (best =   0.819), reward = 0.560241550873
Episode =  629, score = (best =   0.819), reward = 0.559759036419
Episode =  630, score = (best =   0.819), reward = 0.559783680985
Episode =  631, score = (best =   0.819), reward = 0.55976420114
Episode =  632, score = (best =   0.819), reward = 0.559765884894
Episode =  633, score = (best =   0.819), reward = 0.559770462212
Episode =  634, score = (best =   0.819), reward = 0.559758090364
Episode =  635, score = (best =   0.819), reward = 0.559783432562
Episode =  636, score = (best =   0.819), reward = 0.559787529534
Episode =  637, score = (best =   0.819), reward = 0.560239151578
Episode =  638, score = (best =   0.819), reward = 0.560189600339
Episode =  639, score = (best =   0.819), reward = 0.559847402556
Episode =  640, score = (best =   0.819), reward = 0.56024650425
Episode =  641, score = (best =   0.819), reward = 0.559789296797
Episode =  642, score = (best =   0.819), reward = 0.560216904139
Episode =  643, score = (best =   0.819), reward = 0.560173063277
Episode =  644, score = (best =   0.819), reward = 0.559779022552
Episode =  645, score = (best =   0.819), reward = 0.560242589166
Episode =  646, score = (best =   0.819), reward = 0.559751341494
Episode =  647, score = (best =   0.819), reward = 0.560161113895
Episode =  648, score = (best =   0.819), reward = 0.559797507811
Episode =  649, score = (best =   0.819), reward = 0.560254194344
Episode =  650, score = (best =   0.819), reward = 0.559763972958
Episode =  651, score = (best =   0.819), reward = 0.55975306817
Episode =  652, score = (best =   0.819), reward = 0.560255058871
Episode =  653, score = (best =   0.819), reward = 0.560256604972
Episode =  654, score = (best =   0.819), reward = 0.559810201158
Episode =  655, score = (best =   0.819), reward = 0.560264256783
Episode =  656, score = (best =   0.819), reward = 0.560221887047
Episode =  657, score = (best =   0.819), reward = 0.560268000192
Episode =  658, score = (best =   0.819), reward = 0.560253982128
Episode =  659, score = (best =   0.819), reward = 0.560234307719
Episode =  660, score = (best =   0.819), reward = 0.560211442913
Episode =  661, score = (best =   0.819), reward = 0.559763616503
Episode =  662, score = (best =   0.819), reward = 0.560220121232
Episode =  663, score = (best =   0.819), reward = 0.559781696511
Episode =  664, score = (best =   0.819), reward = 0.560233031379
Episode =  665, score = (best =   0.819), reward = 0.559795721484
Episode =  666, score = (best =   0.819), reward = 0.560255458741
Episode =  667, score = (best =   0.819), reward = 0.560257980714
Episode =  668, score = (best =   0.819), reward = 0.560214477522
Episode =  669, score = (best =   0.819), reward = 0.560262736099
Episode =  670, score = (best =   0.819), reward = 0.560211369858
Episode =  671, score = (best =   0.819), reward = 0.560256357792
Episode =  672, score = (best =   0.819), reward = 0.559793581266
Episode =  673, score = (best =   0.819), reward = 0.559749718278
Episode =  674, score = (best =   0.819), reward = 0.560230205303
Episode =  675, score = (best =   0.819), reward = 0.559798715046
Episode =  676, score = (best =   0.819), reward = 0.560263385177
Episode =  677, score = (best =   0.819), reward = 0.559851534211
Episode =  678, score = (best =   0.819), reward = 0.559766339282
Episode =  679, score = (best =   0.819), reward = 0.560213205441
Episode =  680, score = (best =   0.819), reward = 0.560213359652
Episode =  681, score = (best =   0.819), reward = 0.559802343732
Episode =  682, score = (best =   0.819), reward = 0.559754132224
Episode =  683, score = (best =   0.819), reward = 0.559769732453
Episode =  684, score = (best =   0.819), reward = 0.559770938341
Episode =  685, score = (best =   0.819), reward = 0.559774220706
Episode =  686, score = (best =   0.819), reward = 0.560226329775
Episode =  687, score = (best =   0.819), reward = 0.56023888647
Episode =  688, score = (best =   0.819), reward = 0.560192795032
Episode =  689, score = (best =   0.819), reward = 0.560250836479
Episode =  690, score = (best =   0.819), reward = 0.56023037006
Episode =  691, score = (best =   0.819), reward = 0.560075245548
Episode =  692, score = (best =   0.819), reward = 0.560237443275
Episode =  693, score = (best =   0.819), reward = 0.560248978774
Episode =  694, score = (best =   0.819), reward = 0.560186089852
Episode =  695, score = (best =   0.819), reward = 0.560192130131
Episode =  696, score = (best =   0.819), reward = 0.559750218114
Episode =  697, score = (best =   0.819), reward = 0.560221689912
Episode =  698, score = (best =   0.819), reward = 0.560257719982
Episode =  699, score = (best =   0.819), reward = 0.559759208008
Episode =  700, score = (best =   0.819), reward = 0.559775164672
Episode =  701, score = (best =   0.819), reward = 0.55976698883
Episode =  702, score = (best =   0.819), reward = 0.559792809095
Episode =  703, score = (best =   0.819), reward = 0.560240606119
Episode =  704, score = (best =   0.819), reward = 0.559770478245
Episode =  705, score = (best =   0.819), reward = 0.560255646679
Episode =  706, score = (best =   0.819), reward = 0.560236895289
Episode =  707, score = (best =   0.819), reward = 0.560228327316
Episode =  708, score = (best =   0.819), reward = 0.559828228521
Episode =  709, score = (best =   0.819), reward = 0.559756729935
Episode =  710, score = (best =   0.819), reward = 0.560205739342
Episode =  711, score = (best =   0.819), reward = 0.559788791683
Episode =  712, score = (best =   0.819), reward = 0.559793698399
Episode =  713, score = (best =   0.819), reward = 0.559831031817
Episode =  714, score = (best =   0.819), reward = 0.560218557004
Episode =  715, score = (best =   0.819), reward = 0.560245146445
Episode =  716, score = (best =   0.819), reward = 0.559854542302
Episode =  717, score = (best =   0.819), reward = 0.560193216248
Episode =  718, score = (best =   0.819), reward = 0.559752898394
Episode =  719, score = (best =   0.819), reward = 0.55981729762
Episode =  720, score = (best =   0.819), reward = 0.560243124333
Episode =  721, score = (best =   0.819), reward = 0.56022879119
Episode =  722, score = (best =   0.819), reward = 0.5597466438
Episode =  723, score = (best =   0.819), reward = 0.559751371504
Episode =  724, score = (best =   0.819), reward = 0.560249049108
Episode =  725, score = (best =   0.819), reward = 0.559791459911
Episode =  726, score = (best =   0.819), reward = 0.559792187115
Episode =  727, score = (best =   0.819), reward = 0.559779045937
Episode =  728, score = (best =   0.819), reward = 0.560259806845
Episode =  729, score = (best =   0.819), reward = 0.559800794813
Episode =  730, score = (best =   0.819), reward = 0.559753781983
Episode =  731, score = (best =   0.819), reward = 0.560266718114
Episode =  732, score = (best =   0.819), reward = 0.560215157313
Episode =  733, score = (best =   0.819), reward = 0.560255351079
Episode =  734, score = (best =   0.819), reward = 0.560254329872
Episode =  735, score = (best =   0.819), reward = 0.5601717378
Episode =  736, score = (best =   0.819), reward = 0.560275738216
Episode =  737, score = (best =   0.819), reward = 0.559842247266
Episode =  738, score = (best =   0.819), reward = 0.560198926101
Episode =  739, score = (best =   0.819), reward = 0.560273951688
Episode =  740, score = (best =   0.819), reward = 0.560268606349
Episode =  741, score = (best =   0.819), reward = 0.560228971899
Episode =  742, score = (best =   0.819), reward = 0.559772684011
Episode =  743, score = (best =   0.819), reward = 0.560200074012
Episode =  744, score = (best =   0.819), reward = 0.55981983401
Episode =  745, score = (best =   0.819), reward = 0.560244616966
Episode =  746, score = (best =   0.819), reward = 0.560255921067
Episode =  747, score = (best =   0.819), reward = 0.559770791624
Episode =  748, score = (best =   0.819), reward = 0.559748990927
Episode =  749, score = (best =   0.819), reward = 0.56023922154
Episode =  750, score = (best =   0.819), reward = 0.560221614194
Episode =  751, score = (best =   0.819), reward = 0.56024891605
Episode =  752, score = (best =   0.819), reward = 0.56021367531
Episode =  753, score = (best =   0.819), reward = 0.560244023866
Episode =  754, score = (best =   0.819), reward = 0.560190500488
Episode =  755, score = (best =   0.819), reward = 0.560206419863
Episode =  756, score = (best =   0.819), reward = 0.559779622052
Episode =  757, score = (best =   0.819), reward = 0.559789392278
Episode =  758, score = (best =   0.819), reward = 0.560256864114
Episode =  759, score = (best =   0.819), reward = 0.560230046873
Episode =  760, score = (best =   0.819), reward = 0.560246820435
Episode =  761, score = (best =   0.819), reward = 0.559779905256
Episode =  762, score = (best =   0.819), reward = 0.559838573515
Episode =  763, score = (best =   0.819), reward = 0.560245822632
Episode =  764, score = (best =   0.819), reward = 0.560223033953
Episode =  765, score = (best =   0.819), reward = 0.560225807129
Episode =  766, score = (best =   0.819), reward = 0.55975348387
Episode =  767, score = (best =   0.819), reward = 0.559767759871
Episode =  768, score = (best =   0.819), reward = 0.560267368869
Episode =  769, score = (best =   0.819), reward = 0.559766540744
Episode =  770, score = (best =   0.819), reward = 0.560161947005
Episode =  771, score = (best =   0.819), reward = 0.560257574694
Episode =  772, score = (best =   0.819), reward = 0.560255069888
Episode =  773, score = (best =   0.819), reward = 0.560242344351
Episode =  774, score = (best =   0.819), reward = 0.560175767231
Episode =  775, score = (best =   0.819), reward = 0.55979783824
Episode =  776, score = (best =   0.819), reward = 0.559778730193
Episode =  777, score = (best =   0.819), reward = 0.559805667472
Episode =  778, score = (best =   0.819), reward = 0.559758727927
Episode =  779, score = (best =   0.819), reward = 0.559745933544
Episode =  780, score = (best =   0.819), reward = 0.559807750001
Episode =  781, score = (best =   0.819), reward = 0.559748441531
Episode =  782, score = (best =   0.819), reward = 0.559771923271
Episode =  783, score = (best =   0.819), reward = 0.559751291451
Episode =  784, score = (best =   0.819), reward = 0.559763297707
Episode =  785, score = (best =   0.819), reward = 0.56026215036
Episode =  786, score = (best =   0.819), reward = 0.560258802654
Episode =  787, score = (best =   0.819), reward = 0.560151807699
Episode =  788, score = (best =   0.819), reward = 0.559803371685
Episode =  789, score = (best =   0.819), reward = 0.560266229473
Episode =  790, score = (best =   0.819), reward = 0.560270846781
Episode =  791, score = (best =   0.819), reward = 0.559809283495
Episode =  792, score = (best =   0.819), reward = 0.560163759478
Episode =  793, score = (best =   0.819), reward = 0.560237376872
Episode =  794, score = (best =   0.819), reward = 0.559778977615
Episode =  795, score = (best =   0.819), reward = 0.559779534539
Episode =  796, score = (best =   0.819), reward = 0.559781722243
Episode =  797, score = (best =   0.819), reward = 0.560169382515
Episode =  798, score = (best =   0.819), reward = 0.559779138753
Episode =  799, score = (best =   0.819), reward = 0.559757560704
Episode =  800, score = (best =   0.819), reward = 0.559808594056
Episode =  801, score = (best =   0.819), reward = 0.559760177928
Episode =  802, score = (best =   0.819), reward = 0.560257200058
Episode =  803, score = (best =   0.819), reward = 0.559767195288
Episode =  804, score = (best =   0.819), reward = 0.56022885186
Episode =  805, score = (best =   0.819), reward = 0.559815105572
Episode =  806, score = (best =   0.819), reward = 0.559799887778
Episode =  807, score = (best =   0.819), reward = 0.559758820768
Episode =  808, score = (best =   0.819), reward = 0.559757839805
Episode =  809, score = (best =   0.819), reward = 0.560237464515
Episode =  810, score = (best =   0.819), reward = 0.56022675485
Episode =  811, score = (best =   0.819), reward = 0.560227393105
Episode =  812, score = (best =   0.819), reward = 0.559826764767
Episode =  813, score = (best =   0.819), reward = 0.559802020731
Episode =  814, score = (best =   0.819), reward = 0.560194537218
Episode =  815, score = (best =   0.819), reward = 0.559821782515
Episode =  816, score = (best =   0.819), reward = 0.56026686789
Episode =  817, score = (best =   0.819), reward = 0.560227102985
Episode =  818, score = (best =   0.819), reward = 0.559752842672
Episode =  819, score = (best =   0.819), reward = 0.56026344339
Episode =  820, score = (best =   0.819), reward = 0.560273392721
Episode =  821, score = (best =   0.819), reward = 0.560274449428
Episode =  822, score = (best =   0.819), reward = 0.559752265083
Episode =  823, score = (best =   0.819), reward = 0.559813385089
Episode =  824, score = (best =   0.819), reward = 0.56021409773
Episode =  825, score = (best =   0.819), reward = 0.559750940703
Episode =  826, score = (best =   0.819), reward = 0.560165419264
Episode =  827, score = (best =   0.819), reward = 0.559749558197
Episode =  828, score = (best =   0.819), reward = 0.560260176208
Episode =  829, score = (best =   0.819), reward = 0.560264327243
Episode =  830, score = (best =   0.819), reward = 0.560263359996
Episode =  831, score = (best =   0.819), reward = 0.560221420088
Episode =  832, score = (best =   0.819), reward = 0.560172233104
Episode =  833, score = (best =   0.819), reward = 0.559773776455
Episode =  834, score = (best =   0.819), reward = 0.560256784952
Episode =  835, score = (best =   0.819), reward = 0.559750896098
Episode =  836, score = (best =   0.819), reward = 0.560172090451
Episode =  837, score = (best =   0.819), reward = 0.560231418968
Episode =  838, score = (best =   0.819), reward = 0.56021253978
Episode =  839, score = (best =   0.819), reward = 0.559784031262
Episode =  840, score = (best =   0.819), reward = 0.559755781584
Episode =  841, score = (best =   0.819), reward = 0.559810335553
Episode =  842, score = (best =   0.819), reward = 0.560209929754
Episode =  843, score = (best =   0.819), reward = 0.560204029914
Episode =  844, score = (best =   0.819), reward = 0.559754538721
Episode =  845, score = (best =   0.819), reward = 0.559796597626
Episode =  846, score = (best =   0.819), reward = 0.559754173208
Episode =  847, score = (best =   0.819), reward = 0.559809123893
Episode =  848, score = (best =   0.819), reward = 0.560259015031
Episode =  849, score = (best =   0.819), reward = 0.559900118642
Episode =  850, score = (best =   0.819), reward = 0.560254101442
Episode =  851, score = (best =   0.819), reward = 0.559843600856
Episode =  852, score = (best =   0.819), reward = 0.559760821262
Episode =  853, score = (best =   0.819), reward = 0.560244836234
Episode =  854, score = (best =   0.819), reward = 0.560270448872
Episode =  855, score = (best =   0.819), reward = 0.560203873374
Episode =  856, score = (best =   0.819), reward = 0.55980407174
Episode =  857, score = (best =   0.819), reward = 0.560271948816
Episode =  858, score = (best =   0.819), reward = 0.560267908414
Episode =  859, score = (best =   0.819), reward = 0.560268128294
Episode =  860, score = (best =   0.819), reward = 0.559766159632
Episode =  861, score = (best =   0.819), reward = 0.559768445671
Episode =  862, score = (best =   0.819), reward = 0.560261138978
Episode =  863, score = (best =   0.819), reward = 0.55977134096
Episode =  864, score = (best =   0.819), reward = 0.560254750132
Episode =  865, score = (best =   0.819), reward = 0.559779741091
Episode =  866, score = (best =   0.819), reward = 0.55977444758
Episode =  867, score = (best =   0.819), reward = 0.560270887351
Episode =  868, score = (best =   0.819), reward = 0.55981536549
Episode =  869, score = (best =   0.819), reward = 0.560247834181
Episode =  870, score = (best =   0.819), reward = 0.559776147618
Episode =  871, score = (best =   0.819), reward = 0.559774361793
Episode =  872, score = (best =   0.819), reward = 0.560225994514
Episode =  873, score = (best =   0.819), reward = 0.559761850815
Episode =  874, score = (best =   0.819), reward = 0.560169832535
Episode =  875, score = (best =   0.819), reward = 0.559760453623
Episode =  876, score = (best =   0.819), reward = 0.5602636555
Episode =  877, score = (best =   0.819), reward = 0.560133359096
Episode =  878, score = (best =   0.819), reward = 0.559771864169
Episode =  879, score = (best =   0.819), reward = 0.560196594494
Episode =  880, score = (best =   0.819), reward = 0.560228050323
Episode =  881, score = (best =   0.819), reward = 0.559753670531
Episode =  882, score = (best =   0.819), reward = 0.559764373721
Episode =  883, score = (best =   0.819), reward = 0.560272799652
Episode =  884, score = (best =   0.819), reward = 0.560237711307
Episode =  885, score = (best =   0.819), reward = 0.559779362328
Episode =  886, score = (best =   0.819), reward = 0.559769706121
Episode =  887, score = (best =   0.819), reward = 0.559838367684
Episode =  888, score = (best =   0.819), reward = 0.560194188284
Episode =  889, score = (best =   0.819), reward = 0.560266770857
Episode =  890, score = (best =   0.819), reward = 0.560233599215
Episode =  891, score = (best =   0.819), reward = 0.560171297264
Episode =  892, score = (best =   0.819), reward = 0.559756710284
Episode =  893, score = (best =   0.819), reward = 0.560236638468
Episode =  894, score = (best =   0.819), reward = 0.560266399611
Episode =  895, score = (best =   0.819), reward = 0.559762729198
Episode =  896, score = (best =   0.819), reward = 0.55974825445
Episode =  897, score = (best =   0.819), reward = 0.559755968851
Episode =  898, score = (best =   0.819), reward = 0.56022721328
Episode =  899, score = (best =   0.819), reward = 0.559879434062
Episode =  900, score = (best =   0.819), reward = 0.559803427425
Episode =  901, score = (best =   0.819), reward = 0.560157471011
Episode =  902, score = (best =   0.819), reward = 0.560271155615
Episode =  903, score = (best =   0.819), reward = 0.560203230657
Episode =  904, score = (best =   0.819), reward = 0.559792116744
Episode =  905, score = (best =   0.819), reward = 0.560259743737
Episode =  906, score = (best =   0.819), reward = 0.559760973692
Episode =  907, score = (best =   0.819), reward = 0.560268497953
Episode =  908, score = (best =   0.819), reward = 0.560205824102
Episode =  909, score = (best =   0.819), reward = 0.559779470314
Episode =  910, score = (best =   0.819), reward = 0.56020508955
Episode =  911, score = (best =   0.819), reward = 0.560189871271
Episode =  912, score = (best =   0.819), reward = 0.559784636626
Episode =  913, score = (best =   0.819), reward = 0.560266037332
Episode =  914, score = (best =   0.819), reward = 0.559863045808
Episode =  915, score = (best =   0.819), reward = 0.560234564861
Episode =  916, score = (best =   0.819), reward = 0.55980084282
Episode =  917, score = (best =   0.819), reward = 0.559807942506
Episode =  918, score = (best =   0.819), reward = 0.560172808763
Episode =  919, score = (best =   0.819), reward = 0.559837811854
Episode =  920, score = (best =   0.819), reward = 0.560255333479
Episode =  921, score = (best =   0.819), reward = 0.56024739031
Episode =  922, score = (best =   0.819), reward = 0.56018879115
Episode =  923, score = (best =   0.819), reward = 0.559812605688
Episode =  924, score = (best =   0.819), reward = 0.559802929004
Episode =  925, score = (best =   0.819), reward = 0.560207810369
Episode =  926, score = (best =   0.819), reward = 0.559776693059
Episode =  927, score = (best =   0.819), reward = 0.559783365153
Episode =  928, score = (best =   0.819), reward = 0.560213590877
Episode =  929, score = (best =   0.819), reward = 0.56025854617
Episode =  930, score = (best =   0.819), reward = 0.560158521892
Episode =  931, score = (best =   0.819), reward = 0.559753660576
Episode =  932, score = (best =   0.819), reward = 0.559780820424
Episode =  933, score = (best =   0.819), reward = 0.559762013847
Episode =  934, score = (best =   0.819), reward = 0.559847062988
Episode =  935, score = (best =   0.819), reward = 0.55978518741
Episode =  936, score = (best =   0.819), reward = 0.560270974701
Episode =  937, score = (best =   0.819), reward = 0.559786361023
Episode =  938, score = (best =   0.819), reward = 0.560178766349
Episode =  939, score = (best =   0.819), reward = 0.560266132669
Episode =  940, score = (best =   0.819), reward = 0.559754979014
Episode =  941, score = (best =   0.819), reward = 0.560221935449
Episode =  942, score = (best =   0.819), reward = 0.560258527305
Episode =  943, score = (best =   0.819), reward = 0.559752390253
Episode =  944, score = (best =   0.819), reward = 0.560235850876
Episode =  945, score = (best =   0.819), reward = 0.55983579318
Episode =  946, score = (best =   0.819), reward = 0.560259273518
Episode =  947, score = (best =   0.819), reward = 0.559771155923
Episode =  948, score = (best =   0.819), reward = 0.560263614196
Episode =  949, score = (best =   0.819), reward = 0.560260681108
Episode =  950, score = (best =   0.819), reward = 0.559806826911
Episode =  951, score = (best =   0.819), reward = 0.559798099832
Episode =  952, score = (best =   0.819), reward = 0.56024357051
Episode =  953, score = (best =   0.819), reward = 0.559778731487
Episode =  954, score = (best =   0.819), reward = 0.560243345027
Episode =  955, score = (best =   0.819), reward = 0.559799724344
Episode =  956, score = (best =   0.819), reward = 0.56022024701
Episode =  957, score = (best =   0.819), reward = 0.559769490928
Episode =  958, score = (best =   0.819), reward = 0.560265707366
Episode =  959, score = (best =   0.819), reward = 0.55979238234
Episode =  960, score = (best =   0.819), reward = 0.559820670791
Episode =  961, score = (best =   0.819), reward = 0.559784851752
Episode =  962, score = (best =   0.819), reward = 0.560201010404
Episode =  963, score = (best =   0.819), reward = 0.559752488968
Episode =  964, score = (best =   0.819), reward = 0.560235877144
Episode =  965, score = (best =   0.819), reward = 0.560248253104
Episode =  966, score = (best =   0.819), reward = 0.560241776072
Episode =  967, score = (best =   0.819), reward = 0.560214320622
Episode =  968, score = (best =   0.819), reward = 0.559811965276
Episode =  969, score = (best =   0.819), reward = 0.560263145334
Episode =  970, score = (best =   0.819), reward = 0.560211145846
Episode =  971, score = (best =   0.819), reward = 0.560246279561
Episode =  972, score = (best =   0.819), reward = 0.559796987539
Episode =  973, score = (best =   0.819), reward = 0.560211564892
Episode =  974, score = (best =   0.819), reward = 0.559801397731
Episode =  975, score = (best =   0.819), reward = 0.560263833481
Episode =  976, score = (best =   0.819), reward = 0.560251128268
Episode =  977, score = (best =   0.819), reward = 0.55976667642
Episode =  978, score = (best =   0.819), reward = 0.56017941138
Episode =  979, score = (best =   0.819), reward = 0.56019954057
Episode =  980, score = (best =   0.819), reward = 0.560265127468
Episode =  981, score = (best =   0.819), reward = 0.559755891889
Episode =  982, score = (best =   0.819), reward = 0.560269155401
Episode =  983, score = (best =   0.819), reward = 0.559776743402
Episode =  984, score = (best =   0.819), reward = 0.559777148844
Episode =  985, score = (best =   0.819), reward = 0.559795183508
Episode =  986, score = (best =   0.819), reward = 0.560262738029
Episode =  987, score = (best =   0.819), reward = 0.560244978243
Episode =  988, score = (best =   0.819), reward = 0.560201686021
Episode =  989, score = (best =   0.819), reward = 0.560208384187
Episode =  990, score = (best =   0.819), reward = 0.559753541383
Episode =  991, score = (best =   0.819), reward = 0.56024732454
Episode =  992, score = (best =   0.819), reward = 0.560196471529
Episode =  993, score = (best =   0.819), reward = 0.559749160023
Episode =  994, score = (best =   0.819), reward = 0.559762080346
Episode =  995, score = (best =   0.819), reward = 0.559757298974
Episode =  996, score = (best =   0.819), reward = 0.560226108856
Episode =  997, score = (best =   0.819), reward = 0.560265559294
Episode =  998, score = (best =   0.819), reward = 0.559778560437
Episode =  999, score = (best =   0.819), reward = 0.560250171995
Episode = 1000, score = (best =   0.819), reward = 0.560265840167</actor.actor> </pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">## Plot the Rewards[¶](#Plot-the-Rewards) Once you are satisfied with your performance, plot the episode rewards, either from a single run, or averaged over multiple runs.</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [60]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">## TODO: Plot the rewards.</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">graph</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="prompt output_prompt">Out[60]:</div>

<div class="output_text output_subarea output_execute_result">

<pre>[<matplotlib.lines.line2d at="" 0x7f048a2e85c0="">]</matplotlib.lines.line2d></pre>

</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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 )</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">## Reflections[¶](#Reflections) **Question 1**: Describe the task that you specified in `task.py`. How did you design the reward function? **Answer**: I designed the reward function to return a total reward between -1 and 2 if it reached the position on the Z axis over the 3 iterations. If it reached more than 98.7% the target position in the Z axis, I awarded 0.3333333333 more for that iteration to be marked as reached.</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">**Question 2**: Discuss your agent briefly, using the following questions as a guide: * What learning algorithm(s) did you try? What worked best for you? * What was your final choice of hyperparameters (such as $\alpha$, $\gamma$, $\epsilon$, etc.)? * What neural network architecture did you use (if any)? Specify layers, sizes, activation functions, etc. **Answer**: * I really just tried the actor critic method since it is suitable for this project, given the continuous inputs and continuous outputs for the task. * I tried many different combinations of theta and sigma to make my exploration better with a little bit of more noise. At last what worked best for me was theta=0.2 and sigma=0.4\. * For the Actor I used the following structure: * 32 nodes in the first layer with RelU activation function * 64 nodes in the second layer with RelU activation function and L2 Regularization * Dropout layer of 40% to prevent overfitting * 128 nodes in the second layer with RelU activation function and L2 Regularization * Dropout layer of 40% to prevent overfitting * 256 nodes in the second layer with RelU activation function and L2 Regularization * For the Critic I used the following structure: * 32 nodes in the first layer with RelU activation function * 64 nodes in the second layer with RelU activation function and L2 Regularization * Dropout layer of 40% to prevent overfitting * 128 nodes in the second layer with RelU activation function and L2 Regularization * Dropout layer of 40% to prevent overfitting * 256 nodes in the second layer with RelU activation function and L2 Regularization * Dropout layer of 40% to prevent overfitting</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">**Question 3**: Using the episode rewards plot, discuss how the agent learned over time. * Was it an easy task to learn or hard? * Was there a gradual learning curve, or an aha moment? * How good was the final performance of the agent? (e.g. mean rewards over the last 10 episodes) **Answer**: * It was an easy task to learn because the reward reached a very high point at first but stayed at local minima later. * There was a sudden learning curve after the first episodes and fell after some episodes. * The last 10 episodes were really bad, I'm thinking that it has to do with the learning rate or replay buffer.</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">**Question 4**: Briefly summarize your experience working on this project. You can use the following prompts for ideas. * What was the hardest part of the project? (e.g. getting started, plotting, specifying the task, etc.) * Did you find anything interesting in how the quadcopter or your agent behaved? **Answer**: * I think the hardest part of the project was where to start. I was really amazed at how everything I learned in this course was necessary to finish this project. * It is really interesting to me that if I use a lower value for the learning rate it seems to stay at a local minima. I noticed this when I used 0.001 and 0.0001\. I finished using a learning rate of 0.001 which works good but stays at local minima after achieving a high reward. * My agent also started working better when I increased the replay buffer batch size. I learned that from the video in this chapter and it really explains it quite well. * For me this was the most challenging project I worked on in my life, I'm glad Udacity makes us complete this project because it makes us review everything that we've learned in order to complete it. * I would recommend developing the actor critic method a little more since this was difficult for me to understand. Siraj has a really good video on YouTube explaining it and it made everything more clear to me.</div>

</div>

</div>

</div>

</div>
