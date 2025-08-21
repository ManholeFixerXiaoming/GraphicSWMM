function Y=LID_GNN_calculationYL_cluster65dual_J(lid)
desiredPythonPath = 'Yourpath\Anaconda3\envs\d2l\python.exe';
pyPath = 'Yourpath';
cd('fullpath')

% Check current Python environment
pe = pyenv;

% If the current Python environment is not the desired one, reset it
if ~strcmp(pe.Executable, desiredPythonPath)
    % Clear the current Python environment
    terminate(pyenv);  % Note: MATLAB can only switch Python environments once during runtime, restart is required for another switch

    % Set the new Python environment
    pyenv('Version', desiredPythonPath);
end

% Ensure the Python module path is loaded
if count(py.sys.path, pyPath) == 0
    insert(py.sys.path, int32(0), pyPath);
end

up=readNPY('complex_all_area.npy');
% up=up(1:3:end);
lidsum=sum(reshape(lid,3,length(lid)/3));

% Prepare input and call Python function
% Example input, generating a 1x81 array
if lidsum<up
    result = py.CPM_complex_cluster65dual_J.run_gcn_model(lid);  % Call Python function
else
    result=inf;
end

% Convert Python result to MATLAB double type
Y = double(result);

% Return result
end
