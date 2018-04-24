# python function which runs aerostruct analysis based on dict input

from __future__ import print_function, division  # Python 2/3 compatability
from six import iteritems   
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
        'with_viscous' : True,
        'cg' : np.array([30., 0., 5.]),
        'print_level': 0,
        # default design variables, applied to all surfaces
        'des_vars' : [
            'alpha',
            'wing.thickness_cp',
            'wing.twist_cp',
            'wing.sweep',
            'wing.dihedral',
            'wing.taper',
            'wing.span',
            'tail.thickness_cp',
            'tail.twist_cp',
            'tail.sweep',
            'tail.dihedral',
            'tail.taper',
            'tail.span'
        ],
        'output_vars' : [
            'fuelburn', 
            'CD', 
            'CL', 
            'weight'
        ]  
    }
    surf_list = [
        {
            'name' : 'wing',
            'num_y' : 7,
            'num_x' : 2,
            'wing_type' : 'CRM',
            'CD0' : 0.015,
            'symmetry' : True,
            'num_twist_cp' : 2,
            'num_thickness_cp': 2,
         },
         {
             'name' : 'tail',
             'num_y' : 7,
             'num_x' : 2,
             'span' : 20.,
             'root_chord' : 5.,
             'wing_type' : 'rect',
             'offset' : np.array([50., 0., 5.]),
             'twist_cp' : np.array([-9.5])
         }
    ]   
    prob_dict.update(user_prob_dict)

    # remove 'des_vars' key and value from prob_dict
    des_vars = prob_dict.pop('des_vars')

    if user_surf_list:
       #print('user surf')
       surf_list = user_surf_list

    # remove surface des_vars key/value from surface dicts
    surf_vars = []
    for surf in surf_list:
	surf_vars += surf.pop('des_vars', [])
	for var in surf_vars:
            des_vars.append(surf['name']+'.'+var)

    # check that values in prob_dict and surf_list are the correct ones
    
    # when wrapping from Matlab, an array of a single value will always
    # be converted to a float in Python and not an iterable, which
    # causes problems.



    for surf in surf_list:
    	for key, val in iteritems(surf):
            if (key in iterable_vars) and (not hasattr(val,'__iter__')):
		surf[key] = np.array([val])  # make an ndarray from list

    #print('des_vars',des_vars)
    # Create OASProblem object 
    OASprob = OASProblem(prob_dict)

#    # Add design variables
#    # problem-specific design vars...
#    prob_des_vars = ['alpha']
#    # surface-specific design vars...
#    surf_des_vars = ['thickness_cp','twist_cp']

    # Add surfaces and surface design vars to OASProblem
    for surf in surf_list:
        #print(surf)
	#if 'twist_cp'
        OASprob.add_surface(surf)
    
    for var in des_vars:
        OASprob.add_desvar(var)

    # setup OpenMDAO components in OASProblem
    OASprob.setup()

    # Note: change to prob.set_solver_print(level=0) when upgrading to OpenMDAO >= 2.0.0
    #OASprob.prob.print_all_convergence(level=0)   # don't print any info during evaluation

    return OASprob

def OAS_run(user_des_vars={}, OASprob=None, *args, **kwargs):
    if not OASprob:
        #print('setup OAS')
        OASprob = OAS_setup()

    # set print option
    #iprint = kwargs.get('iprint',0)  # set default to only print errors and convergence failures

    # set design variables
    if user_des_vars:
        for var, value in iteritems(user_des_vars):
            #print('$$$$$$      var=',var,'  value=',value)

            if not hasattr(value,'flat'):
		value = np.array([value])  # make an ndarray from list
            OASprob.prob[var] = value
    #print('run OAS')
    
    
    
    OASprob.run()
    #print('after run OAS') 
    
    output = OrderedDict()

    # add input variables to output dictionary
    # input_vars = set(user_des_vars) + set(OASprob.prob.
    for item in OASprob.prob.driver._desvars:
        output[item] = OASprob.prob[item]

    # get overall output variables and constraints, return None if not there
    overall_vars = ['fuelburn','CD','CL','L_equals_W','CM','v','rho','cg','weighted_obj','total_weight']
    for item in overall_vars:
        output[item] = OASprob.prob[item] 
#        print('item=',item)
#        print('OASprob.prob[item]=',OASprob.prob[item])
        
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

if __name__ == "__main__":
    print('--INIT--')

    prob_dict = {
        'type': 'aerostruct',
        'optimize': False,
        'with_viscous': True,
        'cg': np.array([30., 0., 5.]),
        'desvars': [],
	'record_db': True,
	'print_level': 0
    }
    surf_list = [
	{
            'name': 'wing',
	    'num_y': 7,
            'num_x': 3,
	    'wing_type': 'CRM',
            'CD0': 0.015,
            'symmetry': True,
            'num_twist_cp': 2,
            'num_thickness_cp': 2,
            'exact_failure_constraint': True,
	    'chord_cp': np.array([0.5, 0.9, 1.2, 2.7]),
	    'span_cos_spacing': 0.5
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
            'span': 20.
        }
    ]
    OASobj = OAS_setup(prob_dict, surf_list)
    #print('INPUT:')
    #print(OASobj.prob.driver.desvars_of_interest())
    #for key, val in iteritems(OASobj.prob.driver._desvars):
    #    print(key+'=',OASobj.prob[key])
    desvars = {
	'alpha':0.25,
	'wing.twist_cp':0.,
	'wing.thickness_cp':0.1,
	'wing.taper':0.75,
	'wing.dihedral':1.,
	'wing.sweep':1.,
	'wing.span':65.,
	'wing.chord_cp':1.0
    }
    out = OAS_run(desvars,OASobj)

    print('Desvars of interest:')
    print(OASobj.prob.driver.desvars_of_interest())
    # pretty print input
    print('INPUT:')
    for item in OASobj.prob.driver._desvars:
	print(item+' = ',out[item])

    # pretty print output
    print('OUTPUT:')
    # print(OASprob.prob.driver.outputs_of_interest())
    for key, val in iteritems(out):
        print(key+' = ',val)
    print('--END--')
    #print(out)


