function root = pack_project_root()
    root = fileparts(fileparts(mfilename('fullpath')));
end
