function Y = lid_gnn_v3_objective(lid)
desiredPythonPath = 'D:\Anaconda3\envs\d2l\python.exe';
projectRoot = pack_project_root();
pyPath = fullfile(projectRoot, 'surrogate');
pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv('Version', desiredPythonPath);
elseif ~strcmp(pe.Executable, desiredPythonPath)
    error(['MATLAB Python is already loaded as: ', char(pe.Executable), newline, ...
           'Restart MATLAB then run:', newline, ...
           'pyenv(''Version'', ''', desiredPythonPath, ''')']);
end
pyPathObj = py.str(pyPath);
if count(py.sys.path, pyPathObj) == 0
    insert(py.sys.path, int32(0), pyPathObj);
end
up = readNPY(fullfile(projectRoot, 'clustering', 'community_output', 'community_total_area.npy'));
up = double(up(:))';
lid = double(lid);
lidsum = sum(reshape(lid, 3, length(lid) / 3), 1);
if all(lidsum <= up + 1e-9)
    module = py.importlib.import_module('cpm_node1020_dual');
    result = module.run_gcn_model(lid);
else
    result = 10000;
end
Y = double(result);
end
