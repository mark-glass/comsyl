
#
# runs using COMSYL-WOFRY "Beamline" adapted for numpy wavefronts ans pickle
#


import numpy
import os

from srxraylib.plot.gol import plot_image, plot

from wofry.propagator.propagator import PropagationElements, PropagationParameters
from syned.beamline.beamline_element import BeamlineElement
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D

from comsyl.parallel.utils import isMaster, barrier
from comsyl.utils.Logger import log
from comsyl.autocorrelation.AutocorrelationFunctionPropagator import AutocorrelationFunctionPropagator


class CWBeamline(object):
    def __init__(self):
        self._beamline_elements = []
        self._propagator_handlers = []
        self._propagator_specific_parameteres = []

    @classmethod
    def initialize_from_propagator_elements_object(cls,propagator_elements_object,propagator_handler="FRESNEL_ZOOM_XY_2D"):
        bl = CWBeamline()
        for i in range(propagator_elements_object.get_propagation_elements_number()):
            bl.add_element(beamline_element=propagator_elements_object.get_propagation_element(i),
                           propagator_handler=propagator_handler,
                           propagator_specific_parameters=propagator_elements_object.get_propagation_element_parameter(i))

        return bl


    def propagation_code(self):
        return "WOFRY"

    def add_element(self,
                    beamline_element=BeamlineElement(),
                    propagator_handler="FRESNEL_ZOOM_XY_2D",
                    propagator_specific_parameters={'shift_half_pixel':1,'magnification_x':1.0,'magnification_y':1.0},
                    ):
        self._beamline_elements.append(beamline_element)
        self._propagator_handlers.append(propagator_handler)
        self._propagator_specific_parameteres.append(propagator_specific_parameters)


    def number_of_elements(self):
        return len(self._beamline_elements)

    def get_beamline_elements(self):
        return self._beamline_elements

    def get_beamline_element(self,index):
        return self.get_beamline_elements()[index]

    def get_propagator_specific_parameters(self):
        return self._propagator_specific_parameteres

    def get_propagator_specific_parameter(self,index):
        return self.get_propagator_specific_parameters()[index]

    def _info(self):

        for i in range(self.number_of_elements()):
            log(">>> element ",i,self._beamline_elements[i])

    def _check_consistency(self):
        assert( len(self._beamline_elements) == len(self._propagator_specific_parameteres) )

    def _check_identical_handlers(self):
        return self._propagator_handlers[1:] == self._propagator_handlers[:-1]



    def propagate(self,wofry_wavefront,mypropagator,return_wavefront_list=True):

        if return_wavefront_list:
            w_out = wofry_wavefront.duplicate()
            output_wavefronts = []
            output_wavefronts.append(wofry_wavefront.duplicate())

            for i in range(self.number_of_elements()):
                w_in = w_out.duplicate()

                #
                # propagating
                #
                #
                propagation_elements = PropagationElements()
                propagation_elements.add_beamline_element(self._beamline_elements[i])

                propagation_parameters = PropagationParameters(wavefront=w_in,
                                                               propagation_elements = propagation_elements)

                propagation_parameters.set_additional_parameters('shift_half_pixel',(self.get_propagator_specific_parameter(i))["shift_half_pixel"])
                propagation_parameters.set_additional_parameters('magnification_x', (self.get_propagator_specific_parameter(i))["magnification_x"])
                propagation_parameters.set_additional_parameters('magnification_y', (self.get_propagator_specific_parameter(i))["magnification_y"])

                w_out = mypropagator.do_propagation(propagation_parameters=propagation_parameters,
                                                        handler_name=self._propagator_handlers[i])

                output_wavefronts.append(w_out)
            return output_wavefronts

        else:

            assert (self._check_identical_handlers()) # chack all handler are identical

            propagation_elements = PropagationElements()
            for i in range(self.number_of_elements()):
                propagation_elements.add_beamline_element(beamline_element=self._beamline_elements[i],
                                                          element_parameters=self.get_propagator_specific_parameter(i))

            propagation_parameters = PropagationParameters(wavefront=wofry_wavefront.duplicate(),
                                                           propagation_elements = propagation_elements)


            w_out = mypropagator.do_propagation(propagation_parameters=propagation_parameters,
                                                handler_name=self._propagator_handlers[0])

            return w_out


    @classmethod
    def propagate_numpy_wavefront(cls,filename_in,filename_out,beamline,mypropagator,return_wavefront_list=True):

        file_content = numpy.load(filename_in)
        e_field = file_content["e_field"]
        coordinates = file_content["coordinates"]
        energies = file_content["energies"]

        x = numpy.linspace(coordinates[0],coordinates[1],e_field.shape[1])
        y = numpy.linspace(coordinates[2],coordinates[3],e_field.shape[2])
        wofry_wf_in = GenericWavefront2D.initialize_wavefront_from_arrays(x,y,e_field[0,:,:,0].copy())
        wofry_wf_in.set_photon_energy(energies[0])


        # wofry_wf_out_list = cls.propagate_classmethod(wofry_wf_in,beamline,mypropagator)
        wofry_wf_out = beamline.propagate(wofry_wf_in,mypropagator,return_wavefront_list=return_wavefront_list)

        if return_wavefront_list:
            wofry_wf_out_list = wofry_wf_out
            wofry_wf_out = wofry_wf_out_list[-1]

        e_field[0,:,:,0] = wofry_wf_out.get_complex_amplitude()

        coordinates[0] = wofry_wf_out.get_coordinate_x()[0]
        coordinates[1] = wofry_wf_out.get_coordinate_x()[-1]
        coordinates[2] = wofry_wf_out.get_coordinate_y()[0]
        coordinates[3] = wofry_wf_out.get_coordinate_y()[-1]

        numpy.savez(filename_out,
                 e_field=e_field,
                 coordinates=coordinates,
                 energies=energies)

        if return_wavefront_list:
            return wofry_wf_out_list
        else:
            return wofry_wf_out

    def propagate_af(self, autocorrelation_function,
                     directory_name="propagation_wofry",
                     af_output_file_root=None,
                     maximum_mode=None,
                     python_to_be_used="/users/srio/OASYS1.1/miniconda3/bin/python"):


        propagator = AutocorrelationFunctionPropagator(self)

        if maximum_mode is None:
            mode_distribution=autocorrelation_function.modeDistribution()
            maximum_mode = mode_distribution[abs(mode_distribution)>0.00005].shape[0]

        propagator.setMaximumMode(maximum_mode)
        data_directory = "%s/%s" % (directory_name, "wavefronts")

        if isMaster():
            if not os.path.exists(directory_name):
                os.mkdir(directory_name)
            if not os.path.exists(data_directory):
                os.mkdir(data_directory)
        barrier()


        # propagated_filename = "%s/%s.npz" % (data_directory, af_name)
        propagated_filename = "%s/%s.npz" % (data_directory, "wavefront")
        af = propagator.propagate(autocorrelation_function, propagated_filename,method="WOFRY",
                                  python_to_be_used=python_to_be_used)

        barrier()
        if isMaster():
            if af_output_file_root is not None:
                af.save("%s.npz" % (af_output_file_root))
                print(">>>>> File written to disk: %s.npz" % (af_output_file_root))

        return af

if __name__ == "__main__":

    import pickle
    from wofry.propagator.propagator import PropagationManager
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D


    #
    # check the propagation of a numpy wavefront
    #

    
    # initialize propagator
    mypropagator = PropagationManager.Instance()
    try:
        mypropagator.add_propagator(FresnelZoomXY2D())
    except:
        log("May be you already initialized propagator and stored FresnelZoomXY2D")

    BEAMLINE = pickle.load(open("/scisoft/data/srio/COMSYL/TESTS/BEAMLINE.p","rb"))

    return_wavefront_list=True

    wf_list = CWBeamline.propagate_numpy_wavefront(
        "/scisoft/data/srio/COMSYL/TESTS/tmp0_hib3-3302_IN.npz",
        "/scisoft/data/srio/COMSYL/TESTS/tmp0_hib3-3302_OUT.npz",
        BEAMLINE,mypropagator,return_wavefront_list=return_wavefront_list)


    if return_wavefront_list:

        print(wf_list)

        plot_image(wf_list[0].get_intensity(),
                   wf_list[0].get_coordinate_x()*1e6,
                   wf_list[0].get_coordinate_y()*1e6,title="IN",)\

        plot_image(wf_list[-1].get_intensity(),
                   wf_list[-1].get_coordinate_x()*1e6,
                   wf_list[-1].get_coordinate_y()*1e6,title="OUT")
    else:

        plot_image(wf_list.get_intensity(),
                   wf_list.get_coordinate_x()*1e6,
                   wf_list.get_coordinate_y()*1e6,title="OUT")