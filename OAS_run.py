# python function which runs aerostruct analysis based on dict input

from __future__ import print_function, division  # Python 2/3 compatability
from six import iteritems, iterkeys
from collections import OrderedDict

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

try:
    from OpenAeroStruct import OASProblem
except:
    from run_classes import OASProblem

import numpy as np

import warnings
warnings.filterwarnings("ignore")


iterable_vars = ['chord_cp','thickness_cp','radius_cp','twist_cp',
                'xshear_cp','yshear_cp','zshear_cp']

def OAS_setup(user_prob_dict={}, user_surf_list=[]):
    # default prob_dict and surf_dict
    prob_dict = {
        'type' : 'aerostruct',
        'optimize' : False,
        # default design variables, applied to all surfaces
        'des_vars' : ['alpha']
    }

    surf_dict = {
        'wing_type': 'CRM',
        'name': 'wing',
        'num_x': 2,
        'num_y': 7,
        'des_vars': ['thickness_cp','twist_cp','xshear_cp',
                    'yshear_cp','zshear_cp','radius_cp',
                    'dihedral','sweep','span','chord_cp','taper']
    }

    surf_list = []
    for surf in user_surf_list:
        surf_list.append(surf_dict.update(surf))

    prob_dict.update(user_prob_dict)

    # remove 'des_vars' key and value from prob_dict
    des_vars = prob_dict.get('des_vars')

    if user_surf_list:
       #print('user surf')
       surf_list = user_surf_list

    # remove surface des_vars key/value from surface dicts
    surf_des_vars = []
    for surf in surf_list:
        surf_des_vars = surf.get('des_vars', [])
        for var in surf_des_vars:
	       des_vars.append(surf['name']+'.'+var)

    # check that values in prob_dict and surf_list are the correct ones

    # when wrapping from Matlab, an array of a single value will always
    # be converted to a float in Python and not an iterable, which
    # causes problems.
    for surf in surf_list:
    	for key, val in iteritems(surf):
            if (key in iterable_vars) and (not hasattr(val,'__iter__')):
		surf[key] = np.array([val])  # make an ndarray from list

    print('des_vars:',des_vars)
    # Create OASProblem object
    OASprob = OASProblem(prob_dict)

    # Add surfaces to OASProblem
    for surf in surf_list:
        OASprob.add_surface(surf)

    # Add design variables to problem
    for var in des_vars:
        OASprob.add_desvar(var)

    # setup OpenMDAO components in OASProblem
    OASprob.setup()

    return OASprob


def OAS_run(user_des_vars={}, OASprob=None, *args, **kwargs):
    if not OASprob:
        OASprob = OAS_setup()

    # set design variables
    if user_des_vars:
        for var, value in iteritems(user_des_vars):
            if not hasattr(value,'flat'):
		        value = np.array([value])  # make an ndarray from list
            OASprob.prob[var] = value

    OASprob.run()

    output = OrderedDict()

    # add input variables to output dictionary
    for item in OASprob.prob.driver._desvars:
        output[item] = OASprob.prob[item]

    # get overall output variables and constraints, return None if not there
    overall_vars = ['fuelburn','CD','CL','L_equals_W','CM','v','rho','cg','weighted_obj','total_weight']
    for item in overall_vars:
        output[item] = OASprob.prob[item]

    # get lifting surface specific variables and constraints, return None if not there
    surface_var_map = {
        'weight' : 'total_perf.<name>structural_weight',
        'CD' : 'total_perf.<name>CD',
        'CL' : 'total_perf.<name>CL',
        'failure' : '<name>perf.failure',
        'vonmises' : '<name>perf.vonmises',
        'thickness_intersects' : '<name>perf.thickness_intersects'
    }

    # lifting surface coupling variables that need trailing "_" removed from surface name
    coupling_var_map = {
        'loads' : 'coupled.<name>.loads',
        'def_mesh' : 'coupled.<name>.def_mesh'
    }

    for surf in OASprob.surfaces:
        for key, val in iteritems(surface_var_map):
            output.update({surf['name']+key:OASprob.prob[val.replace('<name>',surf['name'])]})
        for key, val in iteritems(coupling_var_map):
            output.update({surf['name']+key:OASprob.prob[val.replace('<name>',surf['name'][:-1])]})

    return output


def OAS_run_matlab(*args, **kwargs):
    '''
    OAS_run() with output dict keys changed from '.' to '_' to work with
    matlab struct object
    '''
    output = OAS_run(*args, **kwargs)
    output_keys = list(output.keys())
    for key in output_keys:
        newkey = key.replace('.','_')
        val = output.pop(key)
        output[newkey] = val

    return output


if __name__ == "__main__":
    print('--INIT--')
    prob_dict = {
        'type': 'aerostruct',
        'optimize': False,
        'with_viscous': True,
	    'record_db': True,
	    'print_level': 0,
        'des_vars': ['alpha'],
        'cg': np.array([70., 0., 15.])
    }
    surf_list = [
	   {
            'name': 'wing',
	        'num_y': 13,
            'num_x': 5,
	        'wing_type': 'CRM',
            'CD0': 0.015,
            'symmetry': True,
            'num_twist_cp': 2,
            'num_thickness_cp': 2,
            'exact_failure_constraint': True,
	        'span_cos_spacing': 0.5,
            'des_vars': ['twist_cp','xshear_cp','thickness_cp','twist_cp','xshear_cp',
                        'yshear_cp','zshear_cp','radius_cp','dihedral','sweep','span','chord_cp','taper']
        },
        {
            'name': 'tail',
	        'num_y': 7,
            'num_x': 3,
	        'wing_type': 'rect',
            'exact_failure_constraint': True,
            'root_chord': 5.0,
            'offset': np.array([50., 0., 5.]),
            'twist_cp': np.array([-9.5]),
            'span': 20.,
            'des_vars': ['twist_cp','xshear_cp','thickness_cp','twist_cp','xshear_cp',
                        'yshear_cp','zshear_cp','radius_cp','dihedral','sweep','span','chord_cp','taper']
        }
    ]
    OASobj = OAS_setup(prob_dict, surf_list)
    print('INPUT:')
    print(OASobj.prob.driver.desvars_of_interest())
    #for key, val in iteritems(OASobj.prob.driver._desvars):
    #    print(key+'=',OASobj.prob[key])
    desvars = {
	'alpha':0.25,
	'wing.twist_cp':[0,0],
	'wing.thickness_cp':[0.001,0.2],
	'wing.taper':1,
	'wing.dihedral':1.,
	'wing.sweep':1.,
	'wing.span':65.,
	# 'wing.chord_cp': np.array([0.5, 0.9, 1.]),
    }
    out = OAS_run_matlab(desvars,OASobj)

    print('Desvars of interest:')
    print(OASobj.prob.driver.desvars_of_interest())
    # pretty print input
    print('INPUT:')
    # for item in OASobj.prob.driver._desvars:
	#        print(item+' = ',out[item])

    # pretty print output
    print('OUTPUT:')
    for key, val in iteritems(out):
        print(key+' = ',val)
    print('--END--')
