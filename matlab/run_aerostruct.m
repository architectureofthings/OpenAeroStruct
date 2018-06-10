% Similar to run_aerostruct.py example script

try
    % on Unix systems this setting is required otherwise Matlab crashes
    py.sys.setdlopenflags(int32(10));  % Set RTLD_NOW and RTLD_DEEPBIND
catch
end

% load Python from virtual environment with OpenMDAO 1.7.3 installed
fprintf('Load Python... \n')
[~,~,isloaded] = pyversion;
if ~isloaded
   pyversion 'C:\Users\Sam\repos\OpenAeroStruct\venv\Scripts\python.exe'
end

% add OpenAeroStruct python module to PYTHONPATH
% OAS_PATH = '/general/home/samfriedman';
OAS_PATH = py.os.path.abspath('../..');
P = py.sys.path;
if count(P,OAS_PATH) == 0
    insert(P,int64(0),OAS_PATH);
end

prob_dict = struct;
prob_dict.type = 'aerostruct';
prob_dict.with_viscous = true;
prob_dict.optimize = false;
prob_dict.record_db = false;  % using sqlitedict locks a process
prob_dict.print_level = 0;
prob_dict.alpha = 0.;

% Instantiate problem and add default surface
fprintf('Create OASProblem object with prob_dict... \n');
OAS_prob = py.OpenAeroStruct.run_classes.OASProblem(prob_dict);

% Create a dictionary to store options about the surface
surf_dict = struct;
surf_dict.name = 'wing';
surf_dict.num_y = 7;
surf_dict.num_x = 2;
surf_dict.wing_type = 'CRM';
surf_dict.CD0 = 0.015;
surf_dict.symmetry = true;
surf_dict.num_twist_cp = 2;
surf_dict.num_thickness_cp = 2;
surf_dict.num_chord_cp = 1;
surf_dict.exact_failure_constraint = true;
surf_dict.span_cos_spacing = 0.5;
% % % Add the specified wing surface to the problem
fprintf('Add wing surface to problem... \n');
OAS_prob.add_surface(surf_dict);

%% Add design variables, constraint, and objective on the problem
% OAS_prob.add_desvar('alpha', pyargs('lower',-10., 'upper',10.));
OAS_prob.add_constraint('L_equals_W', pyargs('equals', 0.));
OAS_prob.add_objective('fuelburn', pyargs('scaler', 1e-5));

% Multiple lifting surfaces
surf_dict = struct;
surf_dict.name = 'tail';
surf_dict.num_y = 7;
surf_dict.num_x = 2;
surf_dict.span = 20.;
surf_dict.root_chord = 5.;
surf_dict.wing_type = 'rect';
surf_dict.offset = [50., 0., 5.];
surf_dict.twist_cp = -9.5;
surf_dict.exact_failure_constraint = true;
fprintf('Add tail surface to problem... \n');
OAS_prob.add_surface(surf_dict)

% Add design variables and constraints for both the wing and tail
fprintf('Add design variables and constraints... \n');
OAS_prob.add_desvar('wing.twist_cp', pyargs('lower',-15.,'upper',15.));
OAS_prob.add_desvar('wing.thickness_cp', pyargs('lower',0.01, 'upper',0.25, 'scaler',1e2));
OAS_prob.add_desvar('wing.taper', pyargs('lower',0.2, 'upper',1.5));
OAS_prob.add_desvar('wing.chord_cp', pyargs('lower',0.9, 'upper',1.1));
OAS_prob.add_constraint('wing_perf.failure', pyargs('upper',0.));
OAS_prob.add_constraint('wing_perf.thickness_intersects', pyargs('upper',0.));
OAS_prob.add_constraint('L_equals_W', pyargs('equals', 0.));
OAS_prob.add_objective('fuelburn', pyargs('scaler', 1e-5));
% OAS_prob.add_desvar('tail.twist_cp', pyargs('lower',-15., 'upper',15.));
% OAS_prob.add_desvar('tail.thickness_cp', pyargs('lower',0.01,'upper',0.5,'scaler',1e2));
% OAS_prob.add_constraint('tail_perf.failure', pyargs('upper',0.));
% OAS_prob.add_constraint('tail_perf.thickness_intersects', pyargs('upper',0.));

% Setup problem
fprintf('Set up the problem... \n');

OAS_prob.setup()
fb = OAS_prob.getvar('fuelburn');
% Actually run the problem
fprintf('Run the problem... \n');
tic;
input = {...
    'wing.twist_cp',[12.80374032, 14.73784563],...
    'wing.thickness_cp',[0.03777685, 0.07183272],...
    'wing.taper',0.2,...
    'wing.chord_cp',0.9,...
    'matlab',true
};
output = struct(OAS_prob.run(pyargs(input{:})));
t = toc;
fuelburn = output.fuelburn;

fprintf('\nFuelburn: %.4f \n', fuelburn);
fprintf('Time elapsed: %.4f secs\n', t);
