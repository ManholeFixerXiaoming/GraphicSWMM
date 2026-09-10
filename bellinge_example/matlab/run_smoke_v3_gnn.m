function run_smoke_v3_gnn()
matlabDir = fileparts(mfilename('fullpath'));
addpath(matlabDir);
rng(20260729, 'twister');
assert(exist('platemo', 'file') == 2, 'PlatEMO is not on the MATLAB path.');
assert(exist('pack_project_root', 'file') == 2, 'missing pack_project_root.m');
pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv('Version', 'D:\Anaconda3\envs\d2l\python.exe');
end
platemo( ...
    'algorithm', @NSGAII, ...
    'problem', @problem_cluster_gnn_v3, ...
    'N', 2, ...
    'maxFE', 5, ...
    'save', 0);
fprintf('smoke test finished\n');
end
