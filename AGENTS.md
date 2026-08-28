# RELAY/LAQS

RELAY (or LAQS, I'm undecided on the name) is an algorithm and on-going research idea for finding ideal data layouts for matrices in GPU kernels.

RELAY design is still very much in progress. `notes/relay.tex` contains notes and an overview of the approach. Ideas may lag slightly in the notes or be incomplete, so do not treat it as exact requirements, but rather an implementation guideline.

## Development Guideline

_Structure:_ RELAY is organized as a library with sets of utilities for computing the low-address quotient subspace algorithm. 'bin/laqs.py' defines a CLI utility for actually taking a problem definition, defined in a Python file, and running the solver on it.

_Functionality:_ This is an experimental project. Breaking changes are okay. Please avoid creating clutter and extra code paths to handle backwards compatibility for experimental code.

_Style:_ Use standard python style where available.

_Comments:_ Do not clutter the code with comments. Provide high level comments on sections of code. Document important and non-trivial functions. Where appropriate, leave comments connecting RELAY implementation variables and computation with the algorithm.

_System and Environment (IMPORTANT):_ You are developing on an HPC cluster. The current cluster has MI300A GPUs (actually APUs) and uses the flux job scheduler. Modules can be loaded with `module load ...`. If you are missing rocm, it is likely because it needs to be loaded; this project is currently using rocm 7.0.2 (e.g. `module load rocm/7.0.2`). There is a local python environment in `.venv`. Use this python if ever using python.

If you need access to a GPU, you need to use flux; the login nodes do not have GPUs accessible. You can use `flux run -n1 -g1 -t 5m -q pdebug <my-command>` to run a short command on a GPU. It may not run right away. DO NOT, UNDER ANY CIRCUMSTANCES, EVER USE MORE TASKS THAN 1, MORE GPUS THAN 1, OR A LONGER WALL TIME THAN 5 MINUTES. If you need bigger or longer jobs, then pass back to the user and ask them to run and report the outputs.

If you hit any environment or system software issues, like missing software or inaccessible GPUs, do not spend too long trying to fix them. Try one or two quick fixes if you feel they will work, but do not waste time or tokens on environment issues that the user did not explicitly ask you to fix. More often than not it requires the user to load or install something outside of your harness. So after some quick attempts, please pass the issue back to the user for them to fix. You can describe the issue and possible solutions if it makes sense.

_Filesystems:_ Code directories are generally kept in home: `/g/g16/dnicho`. This filesystem has limited space, however, so only code, metadata, and small data should be kept here. Larger files, such as installs, virtual environments, and data, should be kept in `/usr/WS1/dnicho` which has a lot more space. I tend to create symlinks to /usr/WS1 in home to make things easier. I generally try to match project subdirectory structure across the symlinks, e.g. `/g/g16/dnicho/myprojects/project-foo/.venv` symlinks to `/usr/WS1/dnicho/myprojects/project-foo/.venv/`.

_Building and Installing:_ `.venv` has an editable install, so Python changes don't require re-installing.

_Job Scripts:_ Flux, the system scheduler, copies scripts to a separate temporary directory before execution, so when editing or creating a batch script (i.e. one that will be submitted with `flux batch`) DO NOT use `$0`, `$BASH_SOURCE[0]`, etc. to reference the current file and directory. It will lead to bugs. Either (A) add a check to see if cwd is the expected place (e.g. `if [ -f ./expected-file ]; `) and early exit if it is not or (B) hard-code absolute paths in the script. (A) is preferable alongside a comment saying what the expected CWD of the script is, but (B) is also ok.

## Figures

When helping make plots and figures, here are a couple things to keep in mind:

- Figures should be abundantly clear; most readers will only look at the figures and tables, so they should tell almost the whole story.
- Captions should be self-contained. A reader shouldn't need to look into the text body for help understanding a figure.
- Figures should use markers, hatching, etc. to be color-blind friendly. Bars should have outlines.
- Y-axes should start at zero for units that start at zero, e.g. runtime. Percentages should go from 0 to 100. There are limited exceptions to these rules.
- Titles should be descriptive and not require additional context to understand.
- Text in figures, e.g. labels, axis tick marks, etc., should be large. The smallest text in a figure should be at least as big as the surrounding text in the paper.
- Figure file formats should be PDF, not png or jpg. It is ok to generate png AND PDF as long as PDF is one of them.