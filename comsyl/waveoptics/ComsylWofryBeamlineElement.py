
from syned.syned_object import SynedObject
from syned.beamline.optical_element import OpticalElement
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement

class ComsylWofryBeamlineElement(BeamlineElement):
    def __init__(self, optical_element=None, coordinates=None,
                 propagator_handler="", propagator_specific_parameters=None):

        if optical_element is None:
            optical_element = OpticalElement()
        if coordinates is None:
            coordinates = ElementCoordinates()

        super().__init__(optical_element=optical_element, coordinates=coordinates)

        self._propagator_handler = propagator_handler

        if propagator_specific_parameters is None:
            self._propagator_specific_parameters = []
        else:
            self._propagator_specific_parameters = propagator_specific_parameters

        # support text containg name of variable, help text and unit. Will be stored in self._support_dictionary
        self._set_support_text([
                    ("optical_element",                "Optical Element",                ""),
                    ("coordinates",                    "Element coordinates",            ""),
                    ("propagator_handler",             "Propagator Handlers",            ""),
                    ("propagator_specific_parameters", "Propagator Specific Parameters", ""),
                    ] )

    def get_propagator_handler(self):
        return self._propagator_handler

    def get_propagator_specific_parameters(self):
        return self._propagator_specific_parameters


if __name__ == "__main__":

    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.beamline.optical_elements.ideal_elements.screen import WOScreen
    a = ComsylWofryBeamlineElement(
                optical_element=WOScreen(),
                coordinates=ElementCoordinates(p=0.0,q=28.3,angle_radial=0.0,angle_azimuthal=0.0),
                propagator_handler='FRESNEL_ZOOM_XY_2D',
                propagator_specific_parameters={'shift_half_pixel':1,'magnification_x':8.0,'magnification_y':8.0}
    )


    print(a.info())

    print(a.to_json())

    f = open("tmp.json","w")
    f.write(a.to_json())
    f.close()
    print("File written to disk: tmp.json")

    from json_tools import load_from_json_file

    tmp = load_from_json_file("tmp.json")

    print(tmp)