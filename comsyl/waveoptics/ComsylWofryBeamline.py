import numpy
import os
import pickle


from syned.storage_ring.empty_light_source import EmptyLightSource
from syned.beamline.beamline import Beamline as SynedBeamline
from syned.beamline.beamline_element import BeamlineElement as SynedBeamlineElement

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofry.propagator.propagator import PropagationElements, PropagationParameters

from comsyl.parallel.utils import isMaster, barrier
from comsyl.utils.Logger import log
from comsyl.utils.Logger import logAll

from comsyl.autocorrelation.AutocorrelationFunctionPropagator import AutocorrelationFunctionPropagator

from comsyl.waveoptics.ComsylWofryBeamlineElement import ComsylWofryBeamlineElement



class ComsylWofryBeamline(SynedBeamline):
    def __init__(self):

        super().__init__()


    @classmethod
    def initialize_from_lists(cls,
                              list_with_syned_optical_elements,
                              list_with_syned_coordinates,
                              list_with_wofry_propagator_handlers,
                              list_with_wofry_propagator_specific_parameters,
                              light_source=None):
        bl = ComsylWofryBeamline()
        for i in range(len(list_with_syned_optical_elements)):
            bl.add_element(ComsylWofryBeamlineElement(
                optical_element=list_with_syned_optical_elements[i],
                coordinates=list_with_syned_coordinates[i],
                propagator_handler=list_with_wofry_propagator_handlers[i],
                propagator_specific_parameters=list_with_wofry_propagator_specific_parameters[i]
            ))

        if light_source is None:
            # set empty ligt source
            bl.set_light_source(EmptyLightSource())
        else:
            bl.set_light_source(light_source)

        return bl


    def add_element(self,beamline_element=ComsylWofryBeamlineElement()):

        self.append_beamline_element(beamline_element)


    def get_propagator_handler(self, index):
        return self.get_beamline_element_at(index).get_propagator_handler()


    def get_propagator_specific_parameters(self, index):
        return self.get_beamline_element_at(index).get_propagator_specific_parameters()


    def write_file_pickle(self,filename_pickle):
        pickle.dump(self, open(filename_pickle, "wb"))
        print("File written to disk: %s "%filename_pickle)

    def write_file_json(self,filename_json):
        f = open(filename_json,"w")
        f.write(self.to_json())
        f.close()
        print("File written to disk: %s "%filename_json)

    @classmethod
    def load_pickle(cls,file_pickle):
        return pickle.load(open(file_pickle, "rb"))


    #
    # Now the COMSYL stuff
    #
    def propagate(self, wofry_wavefront, mypropagator, return_wavefront_list=False):
        from srxraylib.plot.gol import plot_image

        # plot_image(wofry_wavefront.get_intensity(),1e6*wofry_wavefront.get_coordinate_x(),1e6*wofry_wavefront.get_coordinate_y(), title="source")

        if return_wavefront_list:
            w_out = wofry_wavefront.duplicate()
            output_wavefronts = []
            output_wavefronts.append(wofry_wavefront.duplicate())

            for i in range(self.get_beamline_elements_number()):
                w_in = w_out.duplicate()


                method = 0 # 0 = old, 1=New

                if method == 1:

                    propagation_elements = PropagationElements()
                    propagation_elements.add_beamline_element(beamline_element=self.get_beamline_element_at(i))

                    propagation_parameters = PropagationParameters(wavefront=w_in,
                                                                   propagation_elements=propagation_elements)
                    tmp = self.get_propagator_specific_parameters(i)

                    propagation_parameters.set_additional_parameters('shift_half_pixel',tmp['shift_half_pixel'])
                    propagation_parameters.set_additional_parameters('magnification_x',tmp['magnification_x'])
                    propagation_parameters.set_additional_parameters('magnification_y',tmp['magnification_y'])
                else:
                    propagation_elements = PropagationElements()
                    propagation_elements.add_beamline_element(beamline_element=self.get_beamline_element_at(i),
                                                              element_parameters=self.get_propagator_specific_parameters(i))
                    propagation_parameters = PropagationParameters(wavefront=w_in,
                                                                   propagation_elements=propagation_elements)

                w_out = mypropagator.do_propagation(propagation_parameters=propagation_parameters,
                                                    handler_name=self.get_propagator_handler(i))

                # plot_image(w_out.get_intensity(),1e6*w_out.get_coordinate_x(),1e6*w_out.get_coordinate_y(),title="oe index: %d"%(i),aspect="auto")
                output_wavefronts.append(w_out)

            return output_wavefronts

        else:
            # check all handler are identical
            for i in range(self.get_beamline_elements_number()):
                assert (self.get_propagator_handler(0) == self.get_propagator_handler(i))

            propagation_elements = PropagationElements()
            for i in range(self.get_beamline_elements_number()):
                propagation_elements.add_beamline_element(beamline_element=self._beamline_elements_list[i],
                                                          element_parameters=self.get_propagator_specific_parameters(i))

            propagation_parameters = PropagationParameters(wavefront=wofry_wavefront.duplicate(),
                                                           propagation_elements=propagation_elements)

            w_out = mypropagator.do_propagation(propagation_parameters=propagation_parameters,
                                                handler_name=self.get_propagator_handler(0))

            return [w_out]

    @classmethod
    def propagate_numpy_wavefront(cls, filename_in, filename_out, beamline, mypropagator, return_wavefront_list=True):

        file_content = numpy.load(filename_in)
        e_field = file_content["e_field"]
        coordinates = file_content["coordinates"]
        energies = file_content["energies"]

        x = numpy.linspace(coordinates[0], coordinates[1], e_field.shape[1])
        y = numpy.linspace(coordinates[2], coordinates[3], e_field.shape[2])
        wofry_wf_in = GenericWavefront2D.initialize_wavefront_from_arrays(x, y, e_field[0, :, :, 0].copy())
        wofry_wf_in.set_photon_energy(energies[0])

        # wofry_wf_out_list = cls.propagate_classmethod(wofry_wf_in,beamline,mypropagator)
        wofry_wf_out = beamline.propagate(wofry_wf_in, mypropagator, return_wavefront_list=return_wavefront_list)

        if return_wavefront_list:
            wofry_wf_out_list = wofry_wf_out
            wofry_wf_out = wofry_wf_out_list[-1]

        e_field[0, :, :, 0] = wofry_wf_out.get_complex_amplitude()

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

    #
    # these methods are the common interface available in ComsylSRWBeamline and ComsylWofryBeamline
    #

    def propagation_code(self):
        return "WOFRY"

    def add_undulator_offset(self, offset, p_or_q="q"):
        if p_or_q == "p":
            self._beamline_elements_list[0].get_coordinates()._p += offset
        elif p_or_q == "q":
            self._beamline_elements_list[0].get_coordinates()._q += offset

    #
    #
    #
    def propagate_af(self, autocorrelation_function,
                     directory_name="propagation_wofry",
                     af_output_file_root=None,
                     maximum_mode=None, # this is indeed the NUMBER OF MODES (maximum index plus one)
                     python_to_be_used="/users/srio/OASYS1.1/miniconda3/bin/python"):

        propagator = AutocorrelationFunctionPropagator(self)

        if maximum_mode is None:
            mode_distribution = autocorrelation_function.modeDistribution()
            maximum_mode = mode_distribution[abs(mode_distribution) > 0.00005].shape[0]

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
        af = propagator.propagate(autocorrelation_function, propagated_filename, method="WOFRY",
                                  python_to_be_used=python_to_be_used)

        barrier()
        if isMaster():
            if af_output_file_root is not None:
                af.save("%s.npz" % (af_output_file_root))
                logAll("File written to disk: %s.npz" % (af_output_file_root))

        return af


if __name__ == "__main__":




    # from syned.beamline.element_coordinates import ElementCoordinates
    # from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen
    #
    #
    # # just to test
    # a = ComsylWofryBeamline.initialize_from_lists(
    #     list_with_syned_optical_elements=[WOScreen(),WOScreen(),WOScreen()],
    #     list_with_syned_coordinates=[
    #                 ElementCoordinates(p=0.0, q=28.3, angle_radial=0.0, angle_azimuthal=0.0),
    #                 ElementCoordinates(p=0.0, q=28.3, angle_radial=0.0, angle_azimuthal=0.0),
    #                 ElementCoordinates(p=0.0, q=28.3, angle_radial=0.0, angle_azimuthal=0.0),],
    #     list_with_wofry_propagator_handlers=['FRESNEL_ZOOM_XY_2D','FRESNEL_ZOOM_XY_2D','FRESNEL_ZOOM_XY_2D',],
    #     list_with_wofry_propagator_specific_parameters=[
    #                 {'shift_half_pixel': 1, 'magnification_x': 8.0, 'magnification_y': 8.0},
    #                 {'shift_half_pixel': 1, 'magnification_x': 8.0, 'magnification_y': 8.0},
    #                 {'shift_half_pixel': 1, 'magnification_x': 8.0, 'magnification_y': 8.0},])
    #
    #
    # print(a.info())
    #
    # print(a.to_json())

    bl = ComsylWofryBeamline.load_pickle("/users/srio/Working/paper-hierarchical/CODE-COMSYL/tmp/tmp0_chac_beamline.p")
    print(bl.info())

    from comsyl.waveoptics.Wavefront import NumpyWavefront

    w0 = NumpyWavefront.load("/users/srio/Working/paper-hierarchical/CODE-COMSYL/tmp/tmp0_chac_in.npz")
    print(w0)


