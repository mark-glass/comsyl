
#
# runs using "WofrySuperBeamline" adapted for numpy wavefronts ans pickle
#


import numpy


from srxraylib.plot.gol import plot_image, plot


from wofry.propagator.propagator import PropagationElements, PropagationParameters
from syned.beamline.beamline_element import BeamlineElement
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from comsyl.utils.Logger import log


class WofrySuperBeamline(object):
    def __init__(self):
        self._beamline_elements = []
        self._propagator_handlers = []
        self._propagator_specific_parameteres = []

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

    def _info(self):

        for i in range(self.number_of_elements()):
            log(">>> element ",i,self._beamline_elements[i])

    def propagate(self,wofry_wavefront,mypropagator):

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

            propagation_parameters = PropagationParameters(wavefront=w_in,propagation_elements = propagation_elements)

            propagation_parameters.set_additional_parameters('shift_half_pixel',(self._propagator_specific_parameteres[i])["shift_half_pixel"])
            propagation_parameters.set_additional_parameters('magnification_x', (self._propagator_specific_parameteres[i])["magnification_x"])
            propagation_parameters.set_additional_parameters('magnification_y', (self._propagator_specific_parameteres[i])["magnification_y"])

            w_out = mypropagator.do_propagation(propagation_parameters=propagation_parameters,
                                                    handler_name=self._propagator_handlers[i])

            output_wavefronts.append(w_out)



        return output_wavefronts

    @classmethod
    def propagate_classmethod(cls,wofry_wavefront,beamline,mypropagator):
        return beamline.propagate(wofry_wavefront,mypropagator)

    @classmethod
    def propagate_numpy_wavefront(cls,filename_in,filename_out,beamline,mypropagator):

        file_content = numpy.load(filename_in)
        e_field = file_content["e_field"]
        coordinates = file_content["coordinates"]
        energies = file_content["energies"]

        x = numpy.linspace(coordinates[0],coordinates[1],e_field.shape[1])
        y = numpy.linspace(coordinates[2],coordinates[3],e_field.shape[2])
        wofry_wf_in = GenericWavefront2D.initialize_wavefront_from_arrays(x,y,e_field[0,:,:,0].copy())
        wofry_wf_in.set_photon_energy(energies[0])


        wofry_wf_out_list = cls.propagate_classmethod(wofry_wf_in,beamline,mypropagator)


        e_field[0,:,:,0] = wofry_wf_out_list[-1].get_complex_amplitude()

        coordinates[0] = wofry_wf_out_list[-1].get_coordinate_x()[0]
        coordinates[1] = wofry_wf_out_list[-1].get_coordinate_x()[-1]
        coordinates[2] = wofry_wf_out_list[-1].get_coordinate_y()[0]
        coordinates[3] = wofry_wf_out_list[-1].get_coordinate_y()[-1]

        numpy.savez(filename_out,
                 e_field=e_field,
                 coordinates=coordinates,
                 energies=energies)


        return wofry_wf_out_list


if __name__ == "__main__":

    import pickle
    from wofry.propagator.propagator import PropagationManager
    from wofry.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D

    # initialize propagator
    mypropagator = PropagationManager.Instance()
    try:
        mypropagator.add_propagator(FresnelZoomXY2D())
    except:
        log("May be you already initialized propagator and stored FresnelZoomXY2D")


    BEAMLINE = pickle.load(open("/scisoft/xop2.4/extensions/shadowvui/shadow3-scripts/HIGHLIGHTS/BEAMLINE.p","rb"))


    wf_list = WofrySuperBeamline.propagate_numpy_wavefront(
        "/scisoft/xop2.4/extensions/shadowvui/shadow3-scripts/HIGHLIGHTS/MARK/tmp-working/tmp0_hib3-3302_IN.npz",
        "/scisoft/xop2.4/extensions/shadowvui/shadow3-scripts/HIGHLIGHTS/MARK/tmp-working/tmp0_hib3-3302_OUT.npz",
        BEAMLINE,mypropagator)


    plot_image(wf_list[0].get_intensity(),
               wf_list[0].get_coordinate_x()*1e6,
               wf_list[0].get_coordinate_y()*1e6,title="IN",)\

    plot_image(wf_list[-1].get_intensity(),
               wf_list[-1].get_coordinate_x()*1e6,
               wf_list[-1].get_coordinate_y()*1e6,title="OUT")