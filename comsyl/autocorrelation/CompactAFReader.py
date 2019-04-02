import h5py
import numpy as np

from comsyl.autocorrelation.AutocorrelationFunction import AutocorrelationFunction, AutocorrelationFunctionIO
from comsyl.autocorrelation.SigmaMatrix import SigmaMatrix
from comsyl.autocorrelation.AutocorrelationInfo import AutocorrelationInfo
from comsyl.mathcomsyl.Twoform import Twoform
from comsyl.waveoptics.Wavefront import NumpyWavefront
from comsyl.autocorrelation.AutocorrelationFunctionIO import undulator_from_numpy_array
from comsyl.mathcomsyl.TwoformVectors import TwoformVectorsEigenvectors, TwoformVectorsWavefronts

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D


class CompactAFReader(object):
    """
    This class is on top of AutocorrelationFunction and serves as interface for oasys-comsyl
    """
    def __init__(self, af=None, data_dict=None, filename=None, h5f=None):
        self._af   = af
        self._data_dict = data_dict
        self._filename = filename
        self._h5f = h5f

    def __del__(self):
        self.close_h5_file()

    @classmethod
    def initialize_from_h5_file(cls,filename):
        data_dict, h5f = cls.loadh5_to_dictionaire(filename)
        af = cls.fromDictionary(data_dict)
        return CompactAFReader(af,data_dict,filename,h5f)


    @classmethod
    def initialize_from_file(cls,filename):
        filename_extension = filename.split('.')[-1]
        try:
            if filename_extension == "h5":
                return cls.initialize_from_h5_file(filename)
            elif filename_extension == "npz":
                data_dict = AutocorrelationFunctionIO.load(filename)
                af = AutocorrelationFunction.fromDictionary(data_dict)
                af._io._setWasFileLoaded(filename)
                return CompactAFReader(af,data_dict,filename)
            elif filename_extension == "npy":
                filename_without_extension = ('.').join(filename.split('.')[:-1])
                filename_with_npz_extension = filename_without_extension+".npz"
                data_dict = AutocorrelationFunctionIO.load(filename_with_npz_extension)
                af = AutocorrelationFunction.fromDictionary(data_dict)
                af._io._setWasFileLoaded(filename)
                return CompactAFReader(af,data_dict,filename_with_npz_extension)
            else:
                raise FileExistsError("Please enter a file with .npy, .npz or .h5 extension")
        except:
            raise FileExistsError("Error reading file")



    @classmethod
    def loadh5_to_dictionaire(cls,filename):
        try:
            h5f = h5py.File(filename,'r')
        except:
            raise Exception("Failed to read h5 file: %s"%filename)

        data_dict = dict()

        for key in h5f.keys():
            if (key !="twoform_4"):
                data_dict[key] = h5f[key].value
            else:
                data_dict[key] = h5f[key] # TwoformVectorsEigenvectors(h5f[key])
        # IMPORTANT: DO NOT CLOSE FILE
        # h5f.close()
        return data_dict, h5f

    @staticmethod
    def fromDictionary(data_dict):

        sigma_matrix = SigmaMatrix.fromNumpyArray(data_dict["sigma_matrix"])
        undulator = undulator_from_numpy_array(data_dict["undulator"])
        detuning_parameter = data_dict["detuning_parameter"][0]
        energy = data_dict["energy"][0]

        electron_beam_energy = data_dict["electron_beam_energy"][0]


        np_wavefront_0=data_dict["wavefront_0"]
        np_wavefront_1=data_dict["wavefront_1"]
        np_wavefront_2=data_dict["wavefront_2"]
        wavefront = NumpyWavefront.fromNumpyArray(np_wavefront_0, np_wavefront_1, np_wavefront_2)

        try:
            np_exit_slit_wavefront_0=data_dict["exit_slit_wavefront_0"]
            np_exit_slit_wavefront_1=data_dict["exit_slit_wavefront_1"]
            np_exit_slit_wavefront_2=data_dict["exit_slit_wavefront_2"]
            exit_slit_wavefront = NumpyWavefront.fromNumpyArray(np_exit_slit_wavefront_0, np_exit_slit_wavefront_1, np_exit_slit_wavefront_2)
        except:
            exit_slit_wavefront = wavefront.clone()

        try:
            weighted_fields = data_dict["weighted_fields"]
        except:
            weighted_fields = None



        srw_wavefront_rx=data_dict["srw_wavefront_rx"][0]
        srw_wavefront_ry=data_dict["srw_wavefront_ry"][0]

        srw_wavefront_drx = data_dict["srw_wavefront_drx"][0]
        srw_wavefront_dry = data_dict["srw_wavefront_dry"][0]

        info_string = str(data_dict["info"])
        info = AutocorrelationInfo.fromString(info_string)


        sampling_factor=data_dict["sampling_factor"][0]
        minimal_size=data_dict["minimal_size"][0]

        beam_energies = data_dict["beam_energies"]

        static_electron_density = data_dict["static_electron_density"]
        coordinates_x = data_dict["twoform_0"]
        coordinates_y = data_dict["twoform_1"]
        diagonal_elements = data_dict["twoform_2"]
        eigenvalues = data_dict["twoform_3"]

        # do not read the big array with modes
        twoform_vectors = None # data_dict["twoform_4"]

        twoform = Twoform(coordinates_x, coordinates_y, diagonal_elements, eigenvalues, twoform_vectors)

        eigenvector_errors = data_dict["twoform_5"]

        twoform.setEigenvectorErrors(eigenvector_errors)

        af = AutocorrelationFunction(sigma_matrix, undulator, detuning_parameter,energy,electron_beam_energy,
                                     wavefront,exit_slit_wavefront,srw_wavefront_rx, srw_wavefront_drx, srw_wavefront_ry, srw_wavefront_dry,
                                     sampling_factor,minimal_size, beam_energies, weighted_fields,
                                     static_electron_density, twoform,
                                     info)

        af._x_coordinates = coordinates_x
        af._y_coordinates = coordinates_y

        af._intensity = diagonal_elements.reshape(len(coordinates_x), len(coordinates_y))

        return af

    def get_filename(self):
        return self._filename

    def get_af(self):
        return self._af

    def close_h5_file(self):
        try:
            self._h5f.close()
        except:
            pass

    def eigenvalues(self):
        return self._af.eigenvalues()

    def eigenvalue(self,mode):
        return self._af.eigenvalue(mode)

    def x_coordinates(self):
        return self._af.xCoordinates()

    def y_coordinates(self):
        return self._af.yCoordinates()

    def spectral_density(self):
        return np.abs(self._af.intensity())

    def reference_electron_density(self):
        return self._af.staticElectronDensity()

    def reference_undulator_radiation(self):
        return self._af.referenceWavefront().intensity_as_numpy()

    def photon_energy(self):
        return self._af.photonEnergy()

    def total_intensity_from_spectral_density(self):
        return self.spectral_density().real.sum()

    def total_intensity(self):
        return (np.absolute(self._af.intensity())).sum()

    def occupation_array(self):
        return self._af.modeDistribution()

    def occupation(self, i_mode):
        return self.occupation_array()[i_mode]

    def occupation_all_modes(self):
        return self.occupation_array().real.sum()

    def cumulated_occupation_array(self):
        return np.cumsum(np.abs(self.occupation_array()))

    def mode(self, i_mode):
        p = self._data_dict["twoform_4"]
        if isinstance(p,h5py._hl.dataset.Dataset):
            try:
                return p[i_mode,:,:]
            except:
                raise Exception("Problem accessing data in h5 file: %s"%self._filename)
        elif isinstance(p,TwoformVectorsEigenvectors):
            try:
                return self._af.Twoform().vector(i_mode) #AllVectors[i_mode,:,:]
            except:
                raise Exception("Problem accessing data in numpy file: %s"%self._filename)
        else:
            raise Exception("Unknown format for mode stokage.")

    def modes(self):
        p = self._data_dict["twoform_4"]
        if isinstance(p,h5py._hl.dataset.Dataset):
            try:
                return self._data_dict["twoform_4"]
            except:
                raise Exception("Problem accessing data in h5 file: %s"%self._filename)
        elif isinstance(p,TwoformVectorsEigenvectors):
            try:
                return self._af.Twoform().allVectors()
            except:
                raise Exception("Problem accessing data in numpy file: %s"%self._filename)
        else:
            raise Exception("Unknown format for mode stokage.")



    def number_of_modes(self):
        return self.eigenvalues().size

    def number_modes(self):
        return self.number_of_modes()

    @property
    def shape(self):
        return (self.number_modes(), self.x_coordinates().size, self.y_coordinates().size)

    def total_intensity_from_modes(self):
        # WARNING: memory hungry as it will go trough the modes
        # intensity = np.zeros_like(self.mode(0))
        #
        # for i_e, eigenvalue in enumerate(self.eigenvalues()):
        #     intensity += eigenvalue * (np.abs(self.mode(i_e))**2)
        # return np.abs(intensity).sum()
        return self.intensity_from_modes().sum()

    def intensity_from_modes(self,max_mode_index=None):
        intensity = np.zeros_like(self.mode(0))
        eigenvalues = self.eigenvalues()
        if max_mode_index is None:
            pass
        else:
            eigenvalues = eigenvalues[0:max_mode_index+1]
        for i_e, eigenvalue in enumerate(eigenvalues):
            intensity += eigenvalue * np.abs(self.mode(i_e))**2
        return np.abs(intensity)


    def keys(self):
        return self._data_dict.keys()

    def info(self,list_modes=False):
        txt = "\n"


        if list_modes:
            percent = 0.0
            txt += "Occupation and max abs value of the mode\n"
            for i_mode in range(self.number_modes()):
                occupation = np.abs(self.occupation(i_mode))
                percent += occupation
                txt += "     %i occupation: %e, accumulated percent: %12.10f\n" % (i_mode, occupation, 100*percent)


        txt += "\n\n\n\nCOMSYL log text: \n"
        txt += str(self._data_dict["info"])

        txt += "\n\n\n\nCOMSYL info: \n"

        txt += "\n\n***********************************************************************************\n"
        txt += "%i modes on the grid \n" % self.number_modes()
        txt += "x: from %e to %e (%d pixels) \n" % (self.x_coordinates().min(), self.x_coordinates().max(), self.x_coordinates().size )
        txt += "y: from %e to %e (%d pixels) \n" % (self.y_coordinates().min(), self.y_coordinates().max(), self.y_coordinates().size )
        txt += "calculated at %f eV\n" % self.photon_energy()
        txt += "total intensity from spectral density with (maybe improper) normalization: %e\n" % self.total_intensity_from_spectral_density()
        txt += "total intensity: %g\n"%self.total_intensity()
        txt += "Occupation of all modes: %g\n"%self.occupation_all_modes()
        txt += "Lower eigenvalue: %g\n" % self.eigenvalue(0)
        txt += "Eigenvalue_0 / Sum_Eigenvalues (coherent fraction): %g\n" % (self.eigenvalue(0) / self.eigenvalues().sum())
        txt += "Number of Photon Energy points %d \n"%(self.photon_energy().size)
        # SLOW:
        # txt += "total intensity from modes: %g\n"%self.total_intensity_from_modes()
        txt += "Approximated number of modes mode to 90 percent occupancy: %d\n"%self.mode_up_to_percent(90.0)
        txt += "Approximated number of modes mode to 95 percent occupancy: %d\n"%self.mode_up_to_percent(95.0)
        txt += "Approximated number of modes mode to 99 percent occupancy: %d\n"%self.mode_up_to_percent(99.0)
        txt += "\n***********************************************************************************\n\n"

        return txt

    def mode_up_to_percent(self,up_to_percent):
        iedge = np.where(self.cumulated_occupation_array()/self.cumulated_occupation_array()[-1] > 1e-2*up_to_percent)
        if len(iedge[0]) == 0:
            return -1
        else:
            return 1+iedge[0][0]

    @classmethod
    def convert_to_h5(cls,filename,filename_out=None,maximum_number_of_modes=None):

        filename_extension = filename.split('.')[-1]

        if filename_extension == "h5" and maximum_number_of_modes is None:
            print("File is already h5: nothing to convert")
            return None

        af = cls.initialize_from_file(filename)

        if filename_out is None:

            filename_without_extension = ('.').join(filename.split('.')[:-1])

            filename_out = filename_without_extension+".h5"

        af._af.saveh5(filename_out,maximum_number_of_modes=maximum_number_of_modes)

        return CompactAFReader(af)

    def write_h5(self,filename,maximum_number_of_modes=None):
        self._af.saveh5(filename,maximum_number_of_modes=maximum_number_of_modes)




    def CSD_in_one_dimension(self,mode_index_max=None):

        if mode_index_max is None:
            mode_index_max = self.number_of_modes() - 1

        for i in range(mode_index_max+1):
            imodeX = self.mode(i)[:,int(0.5*self.shape[2])]
            imodeY = self.mode(i)[int(0.5*self.shape[1]),:]

            if i == 0:
                Wx1x2 =  np.outer( np.conj(imodeX) , imodeX ) * self.eigenvalue(i)
                Wy1y2 =  np.outer( np.conj(imodeY) , imodeY ) * self.eigenvalue(i)
            else:
                Wx1x2 += np.outer( np.conj(imodeX) , imodeX ) * self.eigenvalue(i)
                Wy1y2 += np.outer( np.conj(imodeY) , imodeY ) * self.eigenvalue(i)

        return Wx1x2,Wy1y2

    # def __WW(self,wf):
    #
    #     WF = wf.get_complex_amplitude()
    #     imodeX = WF[:,int(0.5*self.shape[2])]
    #     imodeY = WF[int(0.5*self.shape[1]),:]
    #
    #     Wx1x2 =  np.outer( np.conj(imodeX) , imodeX ) #* self.eigenvalue(i)
    #     Wy1y2 =  np.outer( np.conj(imodeY) , imodeY ) #* self.eigenvalue(i)
    #
    #
    #     return Wx1x2,Wy1y2
    #
    #
    # def CSD_in_one_dimensionBis(self,mode_index_max=None):
    #
    #     if mode_index_max is None:
    #         mode_index_max = self.number_of_modes() - 1
    #
    #     # i = 3
    #     # imodeX = self.mode(i)[:,int(0.5*self.shape[2])]
    #     # imodeY = self.mode(i)[int(0.5*self.shape[1]),:]
    #     # Wx1x2 =  np.outer( np.conj(imodeX) , imodeX ) * self.eigenvalue(i)
    #     # Wy1y2 =  np.outer( np.conj(imodeY) , imodeY ) * self.eigenvalue(i)
    #
    #     for i in range(mode_index_max+1):
    #
    #         wf =  GenericWavefront2D.initialize_wavefront_from_arrays(
    #             self.x_coordinates(),self.y_coordinates(), self.mode(i)  )
    #         wf.set_photon_energy(self.photon_energy())
    #
    #         # WF = wf.get_complex_amplitude()
    #         # imodeX = WF[:,int(0.5*self.shape[2])]
    #         # imodeY = WF[int(0.5*self.shape[1]),:]
    #
    #         tmp1, tmp2 = self.__WW(wf)
    #
    #         if i == 0:
    #             Wx1x2 = tmp1 * self.eigenvalue(i)
    #             Wy1y2 = tmp2 * self.eigenvalue(i)
    #         else:
    #             Wx1x2 += tmp1 * self.eigenvalue(i)
    #             Wy1y2 += tmp2 * self.eigenvalue(i)
    #
    #     return Wx1x2,Wy1y2


    def Wx1x2(self):
        Wx1x2,Wy1y2 = self.CSD_in_one_dimension()
        return Wx1x2

    def Wy1y2(self):
        Wy1y2,Wy1y2 = self.CSD_in_one_dimension()
        return Wy1y2


if __name__ == "__main__":

    filename = "/users/srio/Working/paper-hierarchical/CODE-COMSYL/propagation_wofry_EBS/rediagonalized.npz"
    af = CompactAFReader.initialize_from_file(filename)
    print(af.info())
