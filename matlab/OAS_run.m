function output = OAS_run(desvars, OASobj)

% Convert matlab cell array of design variables into python dict
py_desvars = py.dict;
for i = 1:2:length(desvars)    % desvars must have even length
    update(py_desvars, py.dict(pyargs( desvars{i}, desvars{i+1} ) ) );
end

% run OAS_run.OAS_run function and return python dict output
py_output = py.OAS_run.OAS_run(py_desvars,OASobj);

% convert python dict to matlab struct
output = struct(py_output);
% convert values of struct array from python objects to matlab
fnames = fieldnames(output);
%save('output');
for idx = 1:length(fnames)
    % fprintf('idx: %d fname{idx}: %s \n',[idx,fnames{idx}]);
    val = np2mat(output.(fnames{idx}));
    % fprintf([fnames{idx},'=%f \n'],val)
    % replace constraint arrays with maximum values
    if any(strfind(fnames{idx},'failure')) || any(strfind(fnames{idx},'thickness_intersects'))
	output.(fnames{idx}) = max(val(:));
    else
        output.(fnames{idx}) = val;
    end
end

end
