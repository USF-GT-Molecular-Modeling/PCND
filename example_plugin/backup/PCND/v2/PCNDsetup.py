R""" Angle potentials.

Angles add forces between specified triplets of particles and are typically used to
model chemical angles between two bonds.

By themselves, angles that have been specified in an initial configuration do nothing. Only when you
specify an angle force (i.e. angle.harmonic), are forces actually calculated between the
listed particles.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.PCND import _PCND
import hoomd

import math;
import sys;

class coeff:
    R""" Define angle coefficients.

    The coefficients for all angle force are specified using this class. Coefficients are
    specified per angle type.

    There are two ways to set the coefficients for a particular angle potential.
    The first way is to save the angle potential in a variable and call :py:meth:`set()` directly.
    See below for an example of this.

    The second method is to build the coeff class first and then assign it to the
    angle potential. There are some advantages to this method in that you could specify a
    complicated set of angle potential coefficients in a separate python file and import
    it into your job script.

    Example::

        my_coeffs = hoomd.md.angle.coeff();
        my_angle_force.angle_coeff.set('polymer', k=330.0, r=0.84)
        my_angle_force.angle_coeff.set('backbone', k=330.0, r=0.84)

    """

    ## \internal
    # \brief Initializes the class
    # \details
    # The main task to be performed during initialization is just to init some variables
    # \param self Python required class instance variable
    def __init__(self):
        self.values = {};
        self.default_coeff = {}

    ## \var values
    # \internal
    # \brief Contains the vector of set values in a dictionary

    ## \var default_coeff
    # \internal
    # \brief default_coeff['coeff'] lists the default value for \a coeff, if it is set

    ## \internal
    # \brief Sets a default value for a given coefficient
    # \details
    # \param name Name of the coefficient to for which to set the default
    # \param value Default value to set
    #
    # Some coefficients have reasonable default values and the user should not be burdened with typing them in
    # all the time. set_default_coeff() sets
    def set_default_coeff(self, name, value):
        self.default_coeff[name] = value;

    def set(self, type, **coeffs):
        R""" Sets parameters for angle types.

        Args:
            type (str): Type of angle (or a list of type names)
            coeffs: Named coefficients (see below for examples)

        Calling :py:meth:`set()` results in one or more parameters being set for a angle type. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the angle
        potential you are setting these coefficients for, see the corresponding documentation.

        All possible angle types as defined in the simulation box must be specified before executing run().
        You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
        angle types that do not exist in the simulation. This can be useful in defining a potential field for many
        different types of angles even when some simulations only include a subset.

        Examples::

            my_angle_force.angle_coeff.set('polymer', k=330.0, r0=0.84)
            my_angle_force.angle_coeff.set('backbone', k=1000.0, r0=1.0)
            my_angle_force.angle_coeff.set(['angleA','angleB'], k=100, r0=0.0)

        Note:
            Single parameters can be updated. If both ``k`` and ``r0`` have already been set for a particle type,
            then executing ``coeff.set('polymer', r0=1.0)`` will update the value of ``r0`` and leave the other
            parameters as they were previously set.

        """
        hoomd.util.print_status_line();

        # listify the input
        type = hoomd.util.listify(type)

        for typei in type:
            self.set_single(typei, coeffs);

    ## \internal
    # \brief Sets a single parameter
    def set_single(self, type, coeffs):
        type = str(type);

        # create the type identifier if it hasn't been created yet
        if (not type in self.values):
            self.values[type] = {};

        # update each of the values provided
        if len(coeffs) == 0:
            hoomd.context.msg.error("No coefficents specified\n");
        for name, val in coeffs.items():
            self.values[type][name] = val;

        # set the default values
        for name, val in self.default_coeff.items():
            # don't override a coeff if it is already set
            if not name in self.values[type]:
                self.values[type][name] = val;

    ## \internal
    # \brief Verifies that all values are set
    # \details
    # \param self Python required self variable
    # \param required_coeffs list of required variables
    #
    # This can only be run after the system has been initialized
    def verify(self, required_coeffs):
        # first, check that the system has been initialized
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot verify angle coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getAngleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getAngleData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                hoomd.context.msg.error("Angle type " +str(type) + " not found in angle coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the angle force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                hoomd.context.msg.error("Angle type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single angle potential coefficient
    # \detail
    # \param type Name of angle type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            hoomd.context.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting angle coeff");

        return self.values[type][coeff_name];

    ## \internal
    # \brief Return metadata
    def get_metadata(self):
        return self.values

class harmonic(force._force):
    R""" Harmonic angle potential.

    The command angle.harmonic specifies a harmonic potential energy between every triplet of particles
    with an angle specified between them.

    .. math::

        V(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2

    where :math:`\theta` is the angle between the triplet of particles.

    Coefficients:

    - :math:`\theta_0` - rest angle  ``t0`` (in radians)
    - :math:`k` - potential constant ``k`` (in units of energy/radians^2)

    Coefficients :math:`k` and :math:`\theta_0` must be set for each type of angle in the simulation using the
    method :py:meth:`angle_coeff.set() <coeff.set()>`.

    Examples::

        harmonic = angle.harmonic()
        harmonic.angle_coeff.set('polymer', k=3.0, t0=0.7851)
        harmonic.angle_coeff.set('backbone', k=100.0, t0=1.0)

    """
    def __init__(self):
        hoomd.util.print_status_line();
        print('working angles!');
        # check that some angles are defined
        if hoomd.context.current.system_definition.getAngleData().getNGlobal() == 0:
            hoomd.context.msg.error("No angles are defined.\n");
            raise RuntimeError("Error creating angle forces");

        # initialize the base class
        force._force.__init__(self);

        # setup the coefficient vector
        self.angle_coeff = coeff();

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _PCND.PCNDForceCompute(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _PCND.PCNDForceCompute(hoomd.context.current.system_definition); #Doesn't exist yet

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        self.required_coeffs = ['k', 't0'];

    ## \internal
    # \brief Update coefficients in C++
    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.angle_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getAngleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getAngleData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.angle_coeff.get(type_list[i], name);

            self.cpp_force.setParams(i, coeff_dict['k'], coeff_dict['t0']);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['angle_coeff'] = self.angle_coeff
        return data


