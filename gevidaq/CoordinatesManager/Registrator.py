# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:30:41 2020

@author: Izak de Heer
"""
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage.draw

from ..HamamatsuCam.HamamatsuActuator import CamActuator
from ..NIDAQ.DAQoperator import DAQmission
from . import CoordinateTransformations
from .backend import readRegistrationImages


class GalvoRegistrator:
    def __init__(self, *args, **kwargs):
        self.cam = CamActuator()
        self.cam.initializeCamera()

    def registration(self, grid_points_x=3, grid_points_y=3):
        """
        By default, generate 9 galvo voltage coordinates from (-5,-5) to (5,5),
        take the camera images of these points, return a function matrix that
        transforms camera_coordinates into galvo_coordinates using polynomial transform.

        Parameters
        grid_points_x : TYPE, optional
            DESCRIPTION. The default is 3.
        grid_points_y : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        transformation : TYPE
            DESCRIPTION.

        """
        galvothread = DAQmission()

        x_coords = np.linspace(-10, 10, grid_points_x + 2)[1:-1]
        y_coords = np.linspace(-10, 10, grid_points_y + 2)[1:-1]

        xy_mesh = np.reshape(
            np.meshgrid(x_coords, y_coords), (2, -1), order="F"
        ).transpose()

        galvo_coordinates = xy_mesh
        camera_coordinates = np.zeros((galvo_coordinates.shape))

        for i in range(galvo_coordinates.shape[0]):
            galvothread.sendSingleAnalog("galvosx", galvo_coordinates[i, 0])
            galvothread.sendSingleAnalog("galvosy", galvo_coordinates[i, 1])
            time.sleep(1)

            image = self.cam.SnapImage(0.06)
            plt.imsave(
                os.getcwd()  # TODO fix path
                + "/CoordinatesManager/Registration_Images/2P/image_"
                + str(i)
                + ".png",
                image,
            )

            camera_coordinates[i, :] = readRegistrationImages.gaussian_fitting(
                image
            )

        logging.info("Galvo Coordinate")
        logging.info(galvo_coordinates)
        logging.info("Camera coordinates")
        logging.info(camera_coordinates)
        del galvothread
        self.cam.Exit()

        transformation_cam2galvo = CoordinateTransformations.polynomial2DFit(
            camera_coordinates, galvo_coordinates, order=1
        )

        transformation_galvo2cam = CoordinateTransformations.polynomial2DFit(
            galvo_coordinates, camera_coordinates, order=1
        )

        logging.info("Transformation found for x:")
        logging.info(transformation_cam2galvo[:, :, 0])
        logging.info("Transformation found for y:")
        logging.info(transformation_cam2galvo[:, :, 1])

        logging.info("galvo2cam found for x:")
        logging.info(transformation_galvo2cam[:, :, 0])
        logging.info("galvo2cam found for y:")
        logging.info(transformation_galvo2cam[:, :, 1])

        return transformation_cam2galvo


class DMDRegistator:
    def __init__(self, DMD, *args, **kwargs):
        self.DMD = DMD
        self.cam = CamActuator()
        self.cam.initializeCamera()

    def registration(
        self,
        laser="640",
        grid_points_x=2,
        grid_points_y=3,
        registration_pattern="circle",
    ):
        logging.info(registration_pattern)
        x_coords = np.linspace(0, 768, grid_points_x + 2)[1:-1]
        y_coords = np.linspace(0, 1024, grid_points_y + 2)[1:-1]

        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        x_coords = np.ravel(x_mesh)
        y_coords = np.ravel(y_mesh)

        dmd_coordinates = np.stack((x_coords, y_coords), axis=1)

        camera_coordinates = np.zeros(dmd_coordinates.shape)

        t1 = time.time()
        for i in range(dmd_coordinates.shape[0]):
            x = int(dmd_coordinates[i, 0])
            y = int(dmd_coordinates[i, 1])
            
            # Mask size is an arbitrary number, 
            # it's a size that's neither too small nor too big for the DMD
            mask_size = 75
            if i == 0:
                x0 = 500
                y0 = 500
                
                mask = DMDRegistator.create_registration_image_touching_squares(x0, y0, mask_size/2)
                
                self.DMD.send_data_to_DMD(mask)
                self.DMD.start_projection()
                
                time.sleep(2)

            if registration_pattern == "squares":
                mask = (
                    DMDRegistator.create_registration_image_touching_squares(
                        x, y
                    )
                )
                
                self.DMD.send_data_to_DMD(mask)
                self.DMD.start_projection()
    
                image = self.cam.SnapImage(0.01)
                # plt.imsave(
                #     os.getcwd()  # TODO fix path
                #     + "/CoordinatesManager/Registration_Images/TouchingSquares/image_"
                #     + str(i)
                #     + ".png",
                #     image,
                # )
                camera_coordinates[
                    i, :
                ] = readRegistrationImages.touchingCoordinateFinder(
                    image
                )
                
            else:
                mask = DMDRegistator.create_registration_image_circle(x, y)
                
                self.DMD.send_data_to_DMD(mask)
                self.DMD.start_projection()
    
                image = self.cam.SnapImage(0.01)
                # plt.imsave(
                #     os.getcwd()  # TODO fix path
                #     + "/CoordinatesManager/Registration_Images/TouchingSquares/image_"
                #     + str(i)
                #     + ".png",
                #     image,
                # )
                camera_coordinates[
                    i, :
                ] = readRegistrationImages.circleCoordinateFinder(
                    image
                )

            self.DMD.send_data_to_DMD(mask)
            self.DMD.start_projection()

            image = self.cam.SnapImage(0.01)
            # plt.imsave(
            #     os.getcwd()  # TODO fix path
            #     + "/CoordinatesManager/Registration_Images/TouchingSquares/image_"
            #     + str(i)
            #     + ".png",
            #     image,
            # )
            camera_coordinates[
                i, :
            ] = readRegistrationImages.circleCoordinateFinder(
                image
            )

            self.DMD.stop_projection()

        t2 = time.time()
        t = t2 - t1
        DMDRegistator.save_coordinates(camera_coordinates, t, registration_pattern, laser, "otsu")

        logging.info("DMD coordinates, Otsu's method:")
        logging.info(dmd_coordinates)
        logging.info("Found camera coordinates, Otsu's method:")
        logging.info(camera_coordinates)

        self.DMD.free_memory()
        self.cam.Exit()

        transformation = CoordinateTransformations.polynomial2DFit(
            camera_coordinates, dmd_coordinates, order=1
        )
        logging.info("Transformation found for x, Otsu's method:")
        logging.info(transformation[:, :, 0])
        logging.info("Transformation found for y, Otsu's method:")
        logging.info(transformation[:, :, 1])
        return transformation

    def create_registration_image_touching_squares(x, y, sigma=75):
        # DMD: 1024 x 768 pixels
        array = np.zeros((768, 1024))
        x_grid, y_grid = np.mgrid[0:768, 0:1024]

        square1 = ((x_grid >= x - sigma) & (x_grid < x)         & (y_grid >= y - sigma) & (y_grid < y))
        square2 = ((x_grid >= x)         & (x_grid < x + sigma) & (y_grid >= y)         & (y_grid < y + sigma))  

        array[square1] = 1
        array[square2] = 1

        return array

    def create_registration_image_circle(x, y, sigma=75):
        # DMD: 1024 x 768 pixels
        array = np.zeros((768, 1024))
        x_grid, y_grid = np.mgrid[0:768, 0:1024]

        circle = ((x_grid - x)**2 + (y_grid - y)**2) < sigma**2

        array[circle] = 1
        return array
    
    def save_coordinates(coordinates, t, preparation_mask, laser, method):
        
        logging.info(str(preparation_mask))
        
        if method == "cc":
            path = r"C:/Labsoftware/gevidaq/gevidaq/CoordinatesManager/backend/CoordinateValues/" + str(preparation_mask)+  "/dmd_coordinates_" + str(method)+ str(laser) + ".txt"
        elif method == "otsu":
            path = r"C:/Labsoftware/gevidaq/gevidaq/CoordinatesManager/backend/CoordinateValues/" + str(preparation_mask)+  "/dmd_coordinates_" + str(laser) + ".txt"
        
        with open(path, "a") as myfile:
            myfile.write(str(coordinates) + ", "+ str(t) + ":\n")
            myfile.close()

    # Registration function for the cross-correlation method
    def registration_cc(self,
        laser="640",
        grid_points_x=2,
        grid_points_y=3,
        registration_pattern="squares",):
        

        x_coords = np.linspace(0, 768, grid_points_x + 2)[1:-1]
        y_coords = np.linspace(0, 1024, grid_points_y + 2)[1:-1]

        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        x_coords = np.ravel(x_mesh)
        y_coords = np.ravel(y_mesh)

        dmd_coordinates = np.stack((x_coords, y_coords), axis=1)

        camera_coordinates = np.zeros(dmd_coordinates.shape)

        t1 = time.time()
        for i in range(dmd_coordinates.shape[0]):
            x = int(dmd_coordinates[i, 0])
            y = int(dmd_coordinates[i, 1])

            # Mask size is an arbitrary number, 
            # it's a size that's neither too small nor too big for the DMD
            mask_size = 75
            if i == 0:
                x0 = 500
                y0 = 500
                
                mask = DMDRegistator.create_registration_image_touching_squares(x0, y0, mask_size/2)
                
                self.DMD.send_data_to_DMD(mask)
                self.DMD.start_projection()
                
                time.sleep(2)
            
                
                
            if registration_pattern == "squares":
                mask = DMDRegistator.create_registration_image_touching_squares(x, y, mask_size/2)
                
                self.DMD.send_data_to_DMD(mask)
                self.DMD.start_projection()
                
                image = self.cam.SnapImage(0.01)

                #plt.imsave(
                #    r"C:/Labsoftware/gevidaq/gevidaq/CoordinatesManager/backend/CoordinateValues/image_"
                #    + str(i)
                #   + ".png",
                #   image,
                #)
                camera_coordinates[
                    i, :
                ] = readRegistrationImages.touchingCoordinateFinder_cc(
                    image, 2*mask_size
                )

            elif registration_pattern == "circle":
                mask = DMDRegistator.create_registration_image_circle(x, y, mask_size)
            
                self.DMD.send_data_to_DMD(mask)
                self.DMD.start_projection()

                image = self.cam.SnapImage(0.01)
                #plt.imsave(
                #    r"C:/Labsoftware/gevidaq/gevidaq/CoordinatesManager/backend/CoordinateValues/image_"
                #    + str(i)
                #    + ".png",
                #   image,
                #)
                camera_coordinates[
                    i, :
                ] = readRegistrationImages.circleCoordinateFinder_cc(
                    image,
                    size=mask_size
                )

            self.DMD.stop_projection()

        t2 = time.time()
        t = t2 - t1
        DMDRegistator.save_coordinates(camera_coordinates, t, registration_pattern, laser, "cc")
        
        logging.info("DMD coordinates, CC method:")
        logging.info(dmd_coordinates)
        logging.info("Found camera coordinates, CC method:")
        logging.info(camera_coordinates)

        self.DMD.free_memory()
        self.cam.Exit()

        transformation = CoordinateTransformations.polynomial2DFit(
            camera_coordinates, dmd_coordinates, order=1
        )
        logging.info("Transformation found for x, CC method:")
        logging.info(transformation[:, :, 0])
        logging.info("Transformation found for y, CC method:")
        logging.info(transformation[:, :, 1])
        return transformation