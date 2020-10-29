#!/usr/bin/env python3

import sys
import os

import random
import pygame
import argparse
import numpy as np
import cv2

import threading
import time

import pyaudio
import wave

from os.path import isfile, join
from robotaxi.gui.pygame import QuitRequestedError
from robotaxi.gameplay.entities import (Point, Field, CellType, RobotaxiDirection, ALL_ROBOTAXI_DIRECTIONS, ROBOTAXI_GROW, WALL_WARP)

from skimage.transform import resize
import matplotlib.pyplot as plt

OBSERVER_ID = 1415 # random seed for color sampling
frame_ct = -1
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
extra_frames = 8
done = False

class captureThread (threading.Thread):

    def __init__(self, threadID, name, counter, data_dir, participant, exp_id, prefix):
    
        super(captureThread, self).__init__()
        #threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.prefix = prefix
        self.data_dir = data_dir
        self.participant = participant
        self.exp_id = exp_id
        
    def run(self):        
        
        global frame_ct, extra_frames        
        print ("Starting " + self.name + ' at ' + time.ctime(time.time()))       
        cap = cv2.VideoCapture(0)
        while frame_ct == -1: 
            time.sleep(0.01)
        img_ct = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True: #framestamp
                cv2.imwrite(self.data_dir+'webcam_imgs/'+self.participant+'/'+self.exp_id+'/'+self.prefix+'_'+str(round(time.time()*1000))+'_'+str(frame_ct)+'_'+str(img_ct)+'.jpg', frame)
                img_ct += 1
                if self._stop_event.is_set(): break
                time.sleep(0.01)                
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
        cap.release()
        cv2.destroyAllWindows()
        print ("Exiting " + self.name)
    
    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
        

class audioThread(threading.Thread):

    def __init__(self, threadID, name, counter, data_dir, participant, prefix):
        super(audioThread, self).__init__()
        self._stop_event = threading.Event()
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.prefix = prefix
        self.data_dir = data_dir
        self.participant = participant
        
    def run(self):        
        global frame_ct, extra_frames
        print ("Starting " + self.name)
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second
        filename = "_output.wav"                
        
        frames = []  # Initialize array to store frames
        while frame_ct == -1:
            time.sleep(0.01) 
        p = pyaudio.PyAudio()  # Create an interface to PortAudio
        print('Recording audio...')  

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)  
        
        
        print("started sound recording at ",time.ctime(time.time()))
        stream.start_stream()
        
        timestamp = str(round(time.time()*1000))
        framestamp = frame_ct
        # Store data in chunks 
        
        while frame_ct < 399 + extra_frames - 1:
            #print(frame_ct)
            data = stream.read(chunk)
            frames.append(data)
            if self._stop_event.is_set(): break

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        print('Finished recording')
        # Save the recorded data as a WAV file
        wf = wave.open(self.data_dir+'audio_log/'+self.prefix+'_'+timestamp+'_'+str(framestamp)+filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        print ("Exiting " + self.name)
    
    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()  

def restore_capture(participant_id):
    print('restoring image captures for participant #', participant_id)
    webcam_img_dir = './webcam_imgs'
    webcam_imgs = [join(webcam_img_dir, f) for f in os.listdir(webcam_img_dir) if isfile(join(webcam_img_dir, f))]
    img_frames = {}
    timesteps = {}
    for img in webcam_imgs:
        info = img.strip().split('/')[-1].split('.')[0].split('_')        
        if not info[0] == participant_id: continue
        #print(info)
        frame_num = int(info[-2])
        timesteps[img] = int(info[-3])
        if frame_num in img_frames:
            img_frames[frame_num].append(img)
        else:
            img_frames[frame_num] = [img]
    
    for frame_num in img_frames:
        img_frames[frame_num].sort()

    return img_frames, timesteps

detected_face_frames = None
detection_ct = 0
def detect_face(raw_img):
    global detected_face_frames, detection_ct
    faces = faceCascade.detectMultiScale(
        raw_img,
        scaleFactor=1.1,
        minNeighbors=5
    )    
    face_img = None
    if len(faces) > 0:       
        face_idx = 0
        for idx in range(len(faces)): 
            if faces[idx][-1]*faces[idx][-2] > faces[face_idx][-1]*faces[face_idx][-2]: face_idx = idx
        (xx, yy, ww, hh) = faces[face_idx]
        #print((xx, yy, ww, hh) )
        if detected_face_frames is None:
            detected_face_frames = [[xx, yy, ww, hh]]*10
        else:
            detected_face_frames[detection_ct%10] = [xx, yy, ww, hh]
        detection_ct += 1
        
    else:
        pass #print ("No face detected!")        
    #plt.imshow(face_img)
    #plt.pause(0.01)
    if detected_face_frames is None: (xx, yy, ww, hh) = (284, 121, 164, 164)
    else: (xx, yy, ww, hh) = np.average(detected_face_frames, axis=0)
    face_img = cv2.cvtColor(raw_img[int(yy-5):int(yy+hh+5), int(xx-5):int(xx+ww+5)],cv2.COLOR_BGR2RGB)
    return face_img

class Colors:

    np.random.seed(2019+OBSERVER_ID)

    SCREEN_BACKGROUND = (139, 139, 139)
    WHITE_BACKGROUND = (255, 255, 255)
    SCORE = (120, 100, 84)
    SCORE_GOOD =  (50, 205, 50)
    SCORE_BAD =  (255, 255, 33)
    SCORE_VERY_BAD =  (205, 20, 50)
    SELECTION = (215, 215, 215)

    CELL_TYPE = {
        CellType.WALL: (26, 26, 26),
        CellType.ROBOTAXI_BODY: (82, 154, 255),
        CellType.ROBOTAXI_HEAD: (0, 77, 64),
        CellType.GOOD_FRUIT: np.random.randint(20,150,size=3) ,
        CellType.BAD_FRUIT:  np.random.randint(80,250,size=3),
        CellType.LAVA: np.random.randint(30,155,size=3),
    }

def permutation(lst):   
    if len(lst) == 0: 
        return [] 
    if len(lst) == 1: 
        return [lst] 
    l = [] 

    for i in range(len(lst)): 
       m = lst[i] 
       remLst = lst[:i] + lst[i+1:] 

       for p in permutation(remLst): 
           l.append([m] + p) 
    return l

def assign_selected_colors(proxy):
    np.random.seed(2019+proxy)

    # Chosen from a color blind palette
    predefined_3colors = [
        (216, 27, 96), # Red
        (30, 136, 229), # Blue
        (255, 193, 7) # Yellow
    ]

    permuted_color_assignment = permutation(predefined_3colors)[np.random.randint(6)]
    Colors.CELL_TYPE[CellType.GOOD_FRUIT] = permuted_color_assignment[0]
    Colors.CELL_TYPE[CellType.BAD_FRUIT] = permuted_color_assignment[1]
    Colors.CELL_TYPE[CellType.LAVA] = permuted_color_assignment[2]

class PyGameGUI:
    """ Provides a Robotaxi GUI powered by Pygame. """

    FPS_LIMIT = 1.6
    #CELL_SIZE = 106

    def __init__(self, field_size=8, noise_prob=0.0, num_noise_color=0, original=False, save_frames=False, proxy=0, stationary=False, agent=None):
        pygame.mixer.pre_init(44100, -16, 2, 32)
        pygame.init()
        pygame.mixer.init()
        pygame.mouse.set_visible(False)

        self.punch_sound = pygame.mixer.Sound('sound/punch.wav')
        self.restart = pygame.mixer.Sound('sound/restart.wav')
        self.begin_sound = pygame.mixer.Sound('sound/begin.wav')
        self.good_sound = pygame.mixer.Sound('sound/good.wav')
        self.bad_sound = pygame.mixer.Sound('sound/road_block_crash.wav')
        self.very_bad_sound = pygame.mixer.Sound('sound/car_crash.wav')
        self.screen = None
        self.fps_clock = None
        self.original = original
        self.pause = True
        self.field_size = field_size
        self.time_thresh = 50 # threshold for timestep left to render red and flash
        self.save_frames = save_frames
        self.intermediate_frames = 21
        self.max_time_limit = None
        self.display_time = False
        self.agent = agent
        if self.field_size == 8:
            self.CELL_SIZE = 96
        else:
            self.CELL_SIZE = 96*2//3
        self.flash_background = True
        self.last_reward = 0
        
        self.car_schemes = ["auto_bus","pickup","truck"]
        self.selected_icon_scheme = 0 # default
        self.set_icon_scheme(self.selected_icon_scheme) 
        self.selected = False
        self.stationary = stationary
        
        self.spawn_icon = pygame.transform.scale(pygame.image.load("icon/wave.png"),(self.CELL_SIZE, self.CELL_SIZE))
        self.wall_icon = pygame.transform.scale(pygame.image.load("icon/forest.png"),(self.CELL_SIZE-5, self.CELL_SIZE-5))
        if self.stationary:
            self.good_fruit_icon = pygame.transform.scale(pygame.image.load("icon/cool_man.png"),(self.CELL_SIZE-15, self.CELL_SIZE-15))
            self.bad_fruit_icon = pygame.transform.scale(pygame.image.load("icon/road_block.png"),(self.CELL_SIZE*3//4, self.CELL_SIZE*3//4))
            self.lava_icon = pygame.transform.scale(pygame.image.load("icon/purple_car.png"),(self.CELL_SIZE, self.CELL_SIZE))
            self.small_crash_icon = pygame.transform.scale(pygame.image.load("icon/road_block_broken.png"),(self.CELL_SIZE*3//4, self.CELL_SIZE*3//4))
            self.big_crash_icon = pygame.transform.scale(pygame.image.load("icon/broken_purple_car.png"),(self.CELL_SIZE, self.CELL_SIZE*3//4))
        else:
            self.good_fruit_icon = pygame.transform.scale(pygame.image.load("icon/cool_man.png"),(self.CELL_SIZE-25, self.CELL_SIZE-25))
            self.bad_fruit_icon = pygame.transform.scale(pygame.image.load("icon/road_block.png"),(self.CELL_SIZE*2//3, self.CELL_SIZE*2//3))
            self.lava_icon = pygame.transform.scale(pygame.image.load("icon/purple_car.png"),(self.CELL_SIZE, self.CELL_SIZE))
            self.small_crash_icon = pygame.transform.scale(pygame.image.load("icon/road_block_broken.png"),(self.CELL_SIZE*2//3, self.CELL_SIZE*2//3))
            self.big_crash_icon = pygame.transform.scale(pygame.image.load("icon/broken_purple_car.png"),(self.CELL_SIZE, self.CELL_SIZE*2//3))

        self.reward_icon = pygame.transform.scale(pygame.image.load("icon/dollar.png"),(self.CELL_SIZE//3, self.CELL_SIZE//3))
        self.thought_icon = pygame.transform.scale(pygame.image.load("icon/bubble.png"),(self.CELL_SIZE//2, self.CELL_SIZE//2))
        self.curr_icon = None
        self.frame_data_to_cell_type = {         
            '0': CellType.EMPTY,
            '1': CellType.GOOD_FRUIT,
            '2': CellType.BAD_FRUIT,
            '3': CellType.LAVA,
            '4': CellType.ROBOTAXI_HEAD,
            '5': CellType.ROBOTAXI_BODY,
            '6': CellType.WALL,
        }
        self.object_type_to_reward = {
            CellType.GOOD_FRUIT: 6,
            CellType.BAD_FRUIT: -1,
            CellType.LAVA: -5,
        }
        self.reward_to_object_type = {
            reward: object_type
            for object_type, reward in self.object_type_to_reward.items()
        }

        self.use_premuted_colors = True

        if self.use_premuted_colors:
            assign_selected_colors(proxy)


        self.internal_padding = self.CELL_SIZE // 4

        self.screen_size = ((self.field_size + 6) * self.CELL_SIZE, self.field_size * self.CELL_SIZE)
        
        self.screen = pygame.display.set_mode(self.screen_size) #, flags=pygame.FULLSCREEN|pygame.HWSURFACE)
        self.curr_head = [0,0]
        self.mid = [0,0]
        self.last_head = [0,0]
        self.text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(23*(self.CELL_SIZE/40.0)))
        self.num_font = pygame.font.Font("fonts/gyparody_tf.ttf", int(36*(self.CELL_SIZE/40.0))) 
        self.marker_font =  pygame.font.Font("fonts/OpenSans-Bold.ttf", int(12*(self.CELL_SIZE/40.0)))
        pygame.display.set_caption('Robotaxi')
    
    def set_icon_scheme(self, idx):
        scheme = self.car_schemes[idx]
        self.south = pygame.transform.scale(pygame.image.load("icon/"+scheme+"_south.png"),(self.CELL_SIZE, self.CELL_SIZE-5))
        self.north = pygame.transform.scale(pygame.image.load("icon/"+scheme+"_north.png"),(self.CELL_SIZE, self.CELL_SIZE-5))        
        self.east = pygame.transform.scale(pygame.image.load("icon/"+scheme+"_east.png"),(self.CELL_SIZE, self.CELL_SIZE-5))
        self.west = pygame.transform.flip(self.east,1,0)

    def fill_background(self, cell_coords=None):
        if self.original:
            background_color = Colors.SCREEN_BACKGROUND
        else:
            background_color = Colors.WHITE_BACKGROUND

        if self.last_reward != 0 and cell_coords is None:
            cell_coords = pygame.Rect(
                0,
                0,
                self.field_size*self.CELL_SIZE,
                self.screen_size[1],
            )
        
        if cell_coords is None:
            self.screen.fill(background_color)
        else:
            pygame.draw.rect(self.screen, background_color, cell_coords)

    def faded_flash(self, reward, intermediate_frames):
        cell_coords = pygame.Rect(
            self.field_size*self.CELL_SIZE,
            0,
            self.screen_size[0]-self.field_size*self.CELL_SIZE,
            self.screen_size[1],
        )

        if reward != 0 and reward is not None:
            self.last_reward = reward
            self.faded_color = Colors.CELL_TYPE[self.reward_to_object_type[self.last_reward]]
            pygame.draw.rect(self.screen, self.faded_color, cell_coords)

        elif self.last_reward != 0:
            color_change = tuple(c/(intermediate_frames*3) for c in tuple(p-q for p, q in zip(Colors.WHITE_BACKGROUND, Colors.CELL_TYPE[self.reward_to_object_type[self.last_reward]])))
            self.faded_color = tuple(min(round(p+q),255) for p, q in zip(self.faded_color, color_change))
            pygame.draw.rect(self.screen, self.faded_color, cell_coords)

            if self.faded_color == Colors.WHITE_BACKGROUND:
                self.last_reward = 0

    def render_frame(self, frame_data, robotaxi_direction, reward=None):
        """ Draw the entire game frame. """

        # self.screen.fill(Colors.SCREEN_BACKGROUND)
        self.fill_background()

        if self.original:
            
            # num_font = pygame.font.SysFont("comicsansms", 32)
            num_font = pygame.font.Font("fonts/gyparody_tf.ttf", int(24*(self.CELL_SIZE/40.0)))
            icon_list = [self.good_fruit_icon, self.bad_fruit_icon, self.lava_icon]

            for i in range(len(icon_list)):
                icon_list[i] = pygame.transform.scale(icon_list[i], tuple([int(0.8*x) for x in icon_list[i].get_size()]))

            text_fields = ["+"+str(self.object_type_to_reward[CellType.GOOD_FRUIT]), str(self.object_type_to_reward[CellType.BAD_FRUIT]), str(self.object_type_to_reward[CellType.LAVA])]
            ct = 0
            for text in text_fields:
                ct += 1
                disp_num = num_font.render(text, True, (0, 0, 0))
                self.screen.blit(disp_num, ((self.field_size+3.0)*self.CELL_SIZE-disp_num.get_width(), self.screen_size[1]//2 - (4.5-ct)*self.screen_size[1] // 10))
                self.screen.blit(icon_list[ct-1], ((self.field_size+0.5)*self.CELL_SIZE, self.screen_size[1]//2 - (4.5-ct)*self.screen_size[1] // 10))
                
            
        for x in range(self.field_size):
            for y in range(self.field_size):
                cell_type = self.frame_data_to_cell_type[frame_data[y][x]]
                """ Draw the cell specified by the field coordinates. """
                cell_coords = pygame.Rect(
                    x * self.CELL_SIZE,
                    y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                # if cell_type == CellType.EMPTY:  
                if cell_type == CellType.EMPTY or cell_type == CellType.ROBOTAXI_BODY:               
                    # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                    self.fill_background(cell_coords)
                    
                else:
                    if self.original:
                        if cell_type == CellType.GOOD_FRUIT:
                            cell_coords = pygame.Rect(
                                x * self.CELL_SIZE + 5,
                                y * self.CELL_SIZE + 5,
                                self.CELL_SIZE,
                                self.CELL_SIZE,
                            )
                            self.screen.blit(self.good_fruit_icon, cell_coords)

                        elif cell_type == CellType.BAD_FRUIT:
                            cell_coords = pygame.Rect(
                                x * self.CELL_SIZE + 5,
                                y * self.CELL_SIZE + 20,
                                self.CELL_SIZE*2//3,
                                self.CELL_SIZE*2//3,
                            )
                            self.screen.blit(self.bad_fruit_icon, cell_coords)

                        elif cell_type == CellType.LAVA:
                            self.screen.blit(self.lava_icon, cell_coords)

                        elif cell_type == CellType.ROBOTAXI_HEAD:
                            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                            self.fill_background(cell_coords)

                            if robotaxi_direction == RobotaxiDirection.NORTH:
                                rotated_icon = self.north
                            elif robotaxi_direction == RobotaxiDirection.WEST:
                                rotated_icon = self.west 
                            elif robotaxi_direction == RobotaxiDirection.SOUTH:
                                rotated_icon = self.south
                            else:
                                rotated_icon = self.east
                            
                            self.curr_icon = rotated_icon
                            #self.screen.blit(rotated_icon, cell_coords)
                            self.last_head = self.curr_head
                            self.curr_head = [x,y]

                        elif cell_type == CellType.ROBOTAXI_BODY:
                            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                            # self.screen.blit(self.body_icon, cell_coords)
                            self.fill_background(cell_coords)

                        elif cell_type  == CellType.WALL:
                            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                            self.fill_background(cell_coords)
                            self.screen.blit(self.wall_icon, cell_coords)
                        else:
                            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                            self.fill_background(cell_coords)
                            color = Colors.CELL_TYPE[cell_type]
                            pygame.draw.rect(self.screen, color, cell_coords, 1)
                            internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                            pygame.draw.rect(self.screen, color, internal_square_coords)
                        
                    else:
                        if cell_type == CellType.ROBOTAXI_HEAD or cell_type == CellType.ROBOTAXI_BODY:
                            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                            self.fill_background(cell_coords)
                            color = Colors.CELL_TYPE[cell_type]             
                            radius =  self.CELL_SIZE // 2 
                            cell_center = (x * self.CELL_SIZE+radius, y * self.CELL_SIZE+radius)
                            if cell_type == CellType.ROBOTAXI_HEAD:
                                self.last_head = self.curr_head
                                self.curr_head = [x,y]
                                #pygame.draw.circle(self.screen, color, cell_center, radius//2)          
                            #pygame.draw.circle(self.screen, color, cell_center, radius-self.internal_padding//2)                                         
                     
                        else:
                            color = Colors.CELL_TYPE[cell_type]               
                            if cell_type == CellType.WALL:
                                pygame.draw.rect(self.screen, color, cell_coords, 1)
                            internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                            pygame.draw.rect(self.screen, color, internal_square_coords)
 
    def render_scoreboard(self, score, time_remaining, frame_idx):
        # scores
        frame_idx = min(frame_idx,len(score)-1)
        text = ("Earnings",'$'+str(score[frame_idx]))
        ct = 0                
        disp_text = self.text_font.render(text[0], True, (0, 0, 0))
        disp_num = self.num_font.render(text[1], True, (0, 0, 0))
        
        self.screen.blit(disp_text, ((self.field_size+0.5)*self.CELL_SIZE, self.screen_size[1] // 2 + 0.4*self.CELL_SIZE))

        ct += 1
        cell_coords = pygame.Rect(
            (self.field_size+0.5)*self.CELL_SIZE,
            self.screen_size[1] // 2 + disp_num.get_height(),
            2.5*self.CELL_SIZE,
            disp_num.get_height()-10,
        )

        bar_size = 5
        if score[frame_idx] <= 0:
            bar_coords = pygame.Rect(
                self.screen_size[0] - 1.95*self.CELL_SIZE,
                self.screen_size[1]//2,
                0.8*self.CELL_SIZE,
                -bar_size*score[frame_idx],
            )
        else:
            bar_coords = pygame.Rect(
                self.screen_size[0] - 1.95*self.CELL_SIZE,
                self.screen_size[1]//2-bar_size*score[frame_idx],
                0.8*self.CELL_SIZE,
                bar_size*score[frame_idx],
            )

        if score[frame_idx] - score[frame_idx-1] < 0:
            delta_coords = pygame.Rect(
                self.screen_size[0] - 1.95*self.CELL_SIZE,
                self.screen_size[1]//2-bar_size*score[frame_idx-1],
                0.8*self.CELL_SIZE,
                -bar_size*(score[frame_idx] - score[frame_idx-1]),
            )
        elif score[frame_idx] - score[frame_idx-1] > 0:
            delta_coords = pygame.Rect(
                self.screen_size[0] - 1.95*self.CELL_SIZE,
                self.screen_size[1]//2-bar_size*score[frame_idx],
                0.8*self.CELL_SIZE,
                bar_size*(score[frame_idx] - score[frame_idx-1]),
            )

        x_start = self.screen_size[0] - 2.4*self.CELL_SIZE
        x_end = self.screen_size[0] - 0.9*self.CELL_SIZE
        y = self.screen_size[1]//2

        origin_marker = self.marker_font.render("$0", True, (0, 0, 0))
        positive_marker = self.marker_font.render("+", True, (0, 0, 0))
        negative_marker = self.marker_font.render("-", True, (0, 0, 0))

        pygame.draw.line(self.screen, (0,0,0), (x_start, y), (x_end, y), 4)
        if score[frame_idx] - score[frame_idx-1] == self.object_type_to_reward[CellType.GOOD_FRUIT]:
            pygame.draw.rect(self.screen, Colors.SCORE_GOOD, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)
            pygame.draw.rect(self.screen, Colors.SCORE_GOOD, delta_coords)
        elif score[frame_idx] - score[frame_idx-1] == self.object_type_to_reward[CellType.BAD_FRUIT]:
            pygame.draw.rect(self.screen, Colors.SCORE_BAD, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)
            pygame.draw.rect(self.screen, Colors.SCORE_BAD, delta_coords)
        elif score[frame_idx] - score[frame_idx-1] == self.object_type_to_reward[CellType.LAVA]:
            pygame.draw.rect(self.screen, Colors.SCORE_VERY_BAD, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)
            pygame.draw.rect(self.screen, Colors.SCORE_VERY_BAD, delta_coords)
        else:
            pygame.draw.rect(self.screen, Colors.SCORE, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)

        pygame.draw.rect(self.screen, (0,0,0), bar_coords, 3)
        self.screen.blit(origin_marker, (self.screen_size[0] - 0.8*self.CELL_SIZE , self.screen_size[1] // 2 - 20 ))
        self.screen.blit(positive_marker, (self.screen_size[0] - 2.33*self.CELL_SIZE , -35+self.screen_size[1] // 2))
        self.screen.blit(negative_marker, (self.screen_size[0] - 2.3*self.CELL_SIZE , -10+self.screen_size[1] // 2))
        self.screen.blit(disp_num, ((self.field_size+0.9)*self.CELL_SIZE, self.screen_size[1] // 2 + self.CELL_SIZE))
        
        ct += 2
  
        # text = ("Time Left",str(round(time_remaining/self.FPS_LIMIT)))
        if self.display_time:
            text = ("Time Left",str(int(round(time_remaining/(self.FPS_LIMIT/1.5)))))
            disp_text = self.text_font.render(text[0], True, (0, 0, 0))
            if time_remaining < self.time_thresh and time_remaining%2 == 0:
                disp_num = self.num_font.render(text[1], True, (225, 50, 50))
            else:
                disp_num = self.num_font.render(text[1], True, (50, 205, 50))
            self.screen.blit(disp_text, (self.screen_size[0] - 4.75*self.CELL_SIZE , 35+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct - disp_text.get_height() ))
            ct += 1
            cell_coords = pygame.Rect(
                self.screen_size[0] - 4.5*self.CELL_SIZE ,
                70+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct - disp_num.get_height(),
                2.5*self.CELL_SIZE,
                disp_num.get_height()-10,
            )
            pygame.draw.rect(self.screen, (59,59,59), cell_coords)
            self.screen.blit(disp_num, (self.screen_size[0] - 4.1*self.CELL_SIZE , 65+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct - disp_num.get_height() ))


    def render(self, log_data, hit_wall, hit_wall_poses, direction, score, max_step_limit, exp_id='test', participant=None):
        """ render from data""" 
    
        global frame_ct, extra_frames
        self.fps_clock = pygame.time.Clock()
        last_frame_face_img = None
        playing = False
        err = 0
        total_time_diff = 0
        face_img_total_ct = 0
        aggregated_frame_ct = 0 
        
        if self.max_time_limit is None:
            self.max_time_limit = max_step_limit//(self.FPS_LIMIT*1.06)
        
        """ Select car scheme first """
        if self.original: 
            while not self.selected:
                # self.screen.fill(Colors.SCREEN_BACKGROUND)
                self.fill_background()
                
                small_text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(20*(self.CELL_SIZE/40.0)))  
                description_font =  pygame.font.SysFont("suruma",  int(16*(self.CELL_SIZE/40.0)))
                car_names = ['Amber','Jade','Ruby']     
                disp_text = self.text_font.render("Vehicle Selection", True, (0, 0, 0)) 
                self.screen.blit(disp_text, (40 , self.screen_size[1] // 3 - disp_text.get_height() ))
                
                disp_text = description_font.render("Which autonomous vehicle would you like to lease?", True, (0, 0, 0))    
                self.screen.blit(disp_text, (40 , self.screen_size[1] // 3 + disp_text.get_height()//4 ))
                
                
                
                for scheme_idx in range(len(self.car_schemes)):
                    scheme = self.car_schemes[scheme_idx]
                    car_icon = pygame.transform.scale(pygame.image.load("icon/"+scheme+"_south.png"),(self.CELL_SIZE, self.CELL_SIZE-5))
                    cell_coords = pygame.Rect(
                        self.screen_size[0]//2 - 2*self.CELL_SIZE + scheme_idx*self.CELL_SIZE*2 ,
                        self.screen_size[1] // 2 - self.CELL_SIZE//3,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    )    
                    if scheme_idx == self.selected_icon_scheme:
                        pygame.draw.rect(self.screen, Colors.SELECTION, cell_coords)
                    self.screen.blit(car_icon, cell_coords)
                    name_text = small_text_font.render(car_names[scheme_idx], True, (80, 80, 80))
                    cell_coords = pygame.Rect(
                        self.screen_size[0]//2  - 2*self.CELL_SIZE + self.CELL_SIZE//2 - name_text.get_width()//2 + scheme_idx*self.CELL_SIZE*2 ,
                        self.screen_size[1] // 2 + self.CELL_SIZE*2//3,
                        name_text.get_width(),
                        name_text.get_height(),
                    )
                    self.screen.blit(name_text, cell_coords)
                
                smaller_text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(16*(self.CELL_SIZE/40.0))) 
                disp_text = smaller_text_font.render("Press <Enter> to confirm", True, (90, 90, 90))    
                self.screen.blit(disp_text, (15 + self.screen_size[0] // 2 - disp_text.get_width()// 2 , self.screen_size[1]*2 // 3 + disp_text.get_height()//2 ))
                
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                             self.selected = True
                             #self.pause = False                        
                        elif event.key == pygame.K_RIGHT:
                            self.selected_icon_scheme += 1
                            self.selected_icon_scheme = min(self.selected_icon_scheme,len(self.car_schemes)-1)
                            self.set_icon_scheme(self.selected_icon_scheme) 
                        elif event.key == pygame.K_LEFT:
                            self.selected_icon_scheme -= 1
                            self.selected_icon_scheme = max(self.selected_icon_scheme,0)
                            self.set_icon_scheme(self.selected_icon_scheme) 
                        elif event.key == pygame.K_ESCAPE:
                            raise QuitRequestedError
                    if event.type == pygame.QUIT:
                        done = True
                        raise QuitRequestedError
                pygame.display.update()          
                self.fps_clock.tick(30)
        else:
            imgs = {}
            timesteps = {}
            sound = None
            if not participant is None:
                imgs, timesteps = restore_capture(participant)
                try:
                    audio_dir = 'audio_log/'
                    audio_files =  [join(audio_dir, f) for f in os.listdir(audio_dir) if isfile(join(audio_dir, f))]
                    sound = None
                    for file in audio_files:
                        if file.split('/')[1].split('_')[0] == participant:
                            sound = pygame.mixer.Sound(file)
                            sound_frame =  int(file.split('_')[-2])
                            sound_timestamp =  int(file.split('_')[-3])
                            break
                except:
                    sound = None
        

        self.render_frame(log_data[0], direction[0])
        
        start_text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(42*(self.CELL_SIZE/40.0))) 
        disp_text = start_text_font.render("Press <Space> to Start", True, (220, 220, 220))
        self.screen.blit(disp_text, (self.screen_size[0] // 2 - disp_text.get_width()// 2 , self.screen_size[1] // 2 - disp_text.get_height()//2 ))
        pygame.display.update()            
             
        """ process frame by frame """

        if self.original: 
            total_frames = len(log_data)+extra_frames
        else: 
             delays = 1 #frames
             total_frames = len(log_data)+extra_frames+delays
        
        for frame_id in range(total_frames): 
                        
            
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause = True
                    if event.key == pygame.K_ESCAPE:
                        raise QuitRequestedError
                if event.type == pygame.QUIT:
                    done = True
                    raise QuitRequestedError

            while self.pause:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.pause = False
                            if frame_id == 0:
                                pygame.mixer.music.load("sound/background1.mp3") 
                                pygame.mixer.music.set_volume(0.4)
                                pygame.mixer.music.play(-1,0.0)
                            self.fps_clock.tick(self.FPS_LIMIT)
                        if event.key == pygame.K_ESCAPE:
                            done = True
                            raise QuitRequestedError
                    if event.type == pygame.QUIT:
                        raise QuitRequestedError

            time_remaining = max_step_limit - frame_ct
            
            
            if self.original:
                if frame_id >= len(log_data): 
                    frame_data = log_data[-1]
                else:
                    frame_data = log_data[frame_id]
                frame_ct = frame_id
                
                pygame.display.set_caption(f'Robotaxi [Steps Remaining: {time_remaining:01d}]')

                # transition animation
                x, y = self.curr_head
                reward = 0
                if frame_ct > 0 and frame_ct <= len(log_data):
                    reward = score[frame_ct] - score[frame_ct-1]
                
                if self.last_head == [0,0]:
                    for interpolate_idx in range(self.intermediate_frames-1): 
                        cell_coords = pygame.Rect(
                            x*self.CELL_SIZE,
                            y*self.CELL_SIZE,
                            self.CELL_SIZE,
                            self.CELL_SIZE,
                        )
                        
                        self.screen.blit(self.spawn_icon, cell_coords)
                        self.render_scoreboard(score, time_remaining, frame_ct)
                        pygame.display.update()
                        self.fps_clock.tick(self.FPS_LIMIT*self.intermediate_frames)
                        if self.save_frames:
                            pygame.image.save(self.screen, 'screenshots/original_replay_frame_%s_%05d.jpg' % (exp_id, aggregated_frame_ct))
                            aggregated_frame_ct += 1
                    # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                    self.fill_background(cell_coords)
                else:                     
                    x0, y0 = self.last_head  
                    imm_coords = pygame.Rect(
                        x0*self.CELL_SIZE,
                        y0*self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    )
                    if abs(x-x0) <= 1 and abs(y-y0) <= 1:                    
                        for interpolate_idx in range(1,self.intermediate_frames-1):
                            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, imm_coords)
                            self.fill_background(imm_coords)                          
                            '''
                            thgt_coords = pygame.Rect(
                                        x0*self.CELL_SIZE + (x-x0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames+self.CELL_SIZE//2,
                                        y0*self.CELL_SIZE + (y-y0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames-self.CELL_SIZE//6-2,
                                        self.CELL_SIZE//2+1,
                                        self.CELL_SIZE//2+2,
                            )
                            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, thgt_coords)
                            '''
                            imm_coords = pygame.Rect(
                                        x0*self.CELL_SIZE + (x-x0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames,
                                        y0*self.CELL_SIZE + (y-y0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE,
                            )
                            
                            self.screen.blit(self.curr_icon, imm_coords)
                            '''
                            if frame_id == 0 or frame_id > 0 and frame_id < len(self.agent) and self.agent[frame_id] != self.agent[frame_id-1]:
                                thgt_coords = pygame.Rect(
                                            x0*self.CELL_SIZE + (x-x0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames+self.CELL_SIZE//2,
                                            y0*self.CELL_SIZE + (y-y0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames-self.CELL_SIZE//6,
                                            self.CELL_SIZE//2,
                                            self.CELL_SIZE//2,
                                )                        
                                self.screen.blit(self.thought_icon, thgt_coords)

                                if self.agent[frame_id] == 0: small_icon = self.good_fruit_icon # passenger                                    
                                elif self.agent[frame_id] == 1:  small_icon = self.bad_fruit_icon # roadblock
                                else: small_icon = self.lava_icon # car
                                small_icon = pygame.transform.scale(small_icon, tuple([int(0.35*x) for x in small_icon.get_size()]))
                                thgt_coords = pygame.Rect(
                                            x0*self.CELL_SIZE + (x-x0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames+self.CELL_SIZE//2+8,
                                            y0*self.CELL_SIZE + (y-y0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames-self.CELL_SIZE//6+2,
                                            self.CELL_SIZE//4,
                                            self.CELL_SIZE//4,
                                )
                                self.screen.blit(small_icon, thgt_coords)
                            '''
                            # restore icon from last frame if crashed
                            if interpolate_idx < self.intermediate_frames / 3 * 2:
                                if reward == self.object_type_to_reward[CellType.GOOD_FRUIT]: # good fruit
                                    cell_coords = pygame.Rect(
                                        self.curr_head[0] * self.CELL_SIZE + 5,
                                        self.curr_head[1] * self.CELL_SIZE + 5,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE,
                                    )                           
                                    self.screen.blit(self.good_fruit_icon, cell_coords)
                                elif reward == self.object_type_to_reward[CellType.BAD_FRUIT]:
                                    cell_coords = pygame.Rect(
                                        self.curr_head[0] * self.CELL_SIZE + 5,
                                        self.curr_head[1] * self.CELL_SIZE + 20,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE,
                                    )   
                                    self.screen.blit(self.bad_fruit_icon, cell_coords)
                                elif reward == self.object_type_to_reward[CellType.LAVA]:
                                    cell_coords = pygame.Rect(
                                        self.curr_head[0] * self.CELL_SIZE,
                                        self.curr_head[1] * self.CELL_SIZE,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE,
                                    )      
                                    self.screen.blit(self.lava_icon, cell_coords)
                            else:
                                if reward == self.object_type_to_reward[CellType.GOOD_FRUIT]:
                                    cell_coords = pygame.Rect(
                                        self.curr_head[0] * self.CELL_SIZE + self.CELL_SIZE//2 - self.CELL_SIZE//6,
                                        self.curr_head[1] * self.CELL_SIZE - 10,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE,
                                    )
                                    self.screen.blit(self.reward_icon, cell_coords)
                                elif reward == self.object_type_to_reward[CellType.BAD_FRUIT]:
                                    cell_coords = pygame.Rect(
                                            self.curr_head[0] * self.CELL_SIZE + 5,
                                            self.curr_head[1] * self.CELL_SIZE + 25,
                                            self.CELL_SIZE*2//3,
                                            self.CELL_SIZE*2//3,
                                        )
                                    self.screen.blit(self.small_crash_icon, cell_coords)
                                elif reward == self.object_type_to_reward[CellType.LAVA]:
                                    cell_coords = pygame.Rect(
                                        self.curr_head[0] * self.CELL_SIZE,
                                        self.curr_head[1] * self.CELL_SIZE + 20,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE*2//3,
                                    )
                                    self.screen.blit(self.big_crash_icon, cell_coords)   
                                    
                            self.render_scoreboard(score, time_remaining,frame_ct-1)
                            pygame.display.update()
                            self.fps_clock.tick(self.FPS_LIMIT*self.intermediate_frames)
                            if self.save_frames:
                                pygame.image.save(self.screen, 'screenshots/original_replay_frame_%s_%05d.jpg' % (exp_id, aggregated_frame_ct))
                                aggregated_frame_ct += 1
                    else:
                        time.sleep(2)
                        start_text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(42*(self.CELL_SIZE/40.0))) 
                        disp_text = start_text_font.render("Next Level", True, (220, 220, 220))
                        self.screen.blit(disp_text, (self.screen_size[0] // 2 - disp_text.get_width()// 2 , self.screen_size[1] // 2 - disp_text.get_height()//2 ))
                        pygame.display.update()
                        self.restart.play()
                        time.sleep(1)
                        
                        self.fill_background(imm_coords)
                        imm_coords = pygame.Rect(
                                    x*self.CELL_SIZE,
                                    y*self.CELL_SIZE,
                                    self.CELL_SIZE,
                                    self.CELL_SIZE,
                        )
                        
                        self.screen.blit(self.curr_icon, imm_coords)
                            

                    # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, imm_coords)
                    self.fill_background(imm_coords)
                # final pose
                
                cell_coords = pygame.Rect(
                    x*self.CELL_SIZE,
                    y*self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                '''
                if not self.stationary:
                    
                    thgt_coords = pygame.Rect(
                                x*self.CELL_SIZE+self.CELL_SIZE//2,
                                y*self.CELL_SIZE-self.CELL_SIZE//6-2,
                                self.CELL_SIZE//2+1,
                                self.CELL_SIZE//2+2,
                    )
                    pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, thgt_coords)
                    
                    if frame_id == 0 or frame_id > 1 and frame_id < len(self.agent) and self.agent[frame_id] != self.agent[frame_id-1]:
                        thgt_coords = pygame.Rect(
                                    x*self.CELL_SIZE+self.CELL_SIZE//2,
                                    y*self.CELL_SIZE-self.CELL_SIZE//6,
                                    self.CELL_SIZE//2,
                                    self.CELL_SIZE//2,
                        )                        
                        self.screen.blit(self.thought_icon, thgt_coords)
                        if self.agent[frame_id] == 0: small_icon = self.good_fruit_icon # passenger                                    
                        elif self.agent[frame_id] == 1:  small_icon = self.bad_fruit_icon # roadblock
                        else: small_icon = self.lava_icon # car
                        small_icon = pygame.transform.scale(small_icon, tuple([int(0.35*x) for x in small_icon.get_size()]))
                        thgt_coords = pygame.Rect(
                                    x*self.CELL_SIZE+self.CELL_SIZE//2+8,
                                    y*self.CELL_SIZE-self.CELL_SIZE//6+2,
                                    self.CELL_SIZE//4,
                                    self.CELL_SIZE//4,
                        )
                        self.screen.blit(small_icon, thgt_coords)
                '''
                self.screen.blit(self.curr_icon, cell_coords)  
                
                if frame_ct == 0:
                    if self.stationary: self.restart.play()
                    else: self.begin_sound.play()
                else:
                    
                    if reward == self.object_type_to_reward[CellType.GOOD_FRUIT]:
                        self.good_sound.play()
                        cell_coords = pygame.Rect(
                            self.curr_head[0] * self.CELL_SIZE + self.CELL_SIZE//2 - self.CELL_SIZE//6,
                            self.curr_head[1] * self.CELL_SIZE - 10,
                            self.CELL_SIZE,
                            self.CELL_SIZE,
                        )
                        self.screen.blit(self.reward_icon, cell_coords)
                    elif reward == self.object_type_to_reward[CellType.BAD_FRUIT]:
                        self.bad_sound.play()
                        cell_coords = pygame.Rect(
                                self.curr_head[0] * self.CELL_SIZE + 5,
                                self.curr_head[1] * self.CELL_SIZE + 25,
                                self.CELL_SIZE*2//3,
                                self.CELL_SIZE*2//3,
                            )
                        self.screen.blit(self.small_crash_icon, cell_coords)
                    elif reward == self.object_type_to_reward[CellType.LAVA]:
                        self.very_bad_sound.play()
                        cell_coords = pygame.Rect(
                            self.curr_head[0] * self.CELL_SIZE,
                            self.curr_head[1] * self.CELL_SIZE + 20,
                            self.CELL_SIZE,
                            self.CELL_SIZE*2//3,
                        )
                        self.screen.blit(self.big_crash_icon, cell_coords)
                
                self.render_scoreboard(score, time_remaining,frame_ct)
                pygame.display.update()
                if reward != 0:  self.fps_clock.tick(self.FPS_LIMIT*2) 
                else: self.fps_clock.tick(self.FPS_LIMIT*self.intermediate_frames)  
                if self.save_frames:
                    pygame.image.save(self.screen, 'screenshots/original_replay_frame_%s_%05d.jpg' % (exp_id, aggregated_frame_ct))
                    aggregated_frame_ct += 1
                #self.fps_clock.tick(self.FPS_LIMIT)  
                
                self.render_frame(frame_data, direction[min(frame_ct,len(direction)-1)])
                if frame_ct > len(log_data):
                    start_text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(62*(self.CELL_SIZE/40.0))) 
                    disp_text = start_text_font.render("Round Finished", True, (220, 220, 220))
                    self.screen.blit(disp_text, (self.screen_size[0] // 2 - disp_text.get_width()// 2 , self.screen_size[1] // 2 - disp_text.get_height()//2 ))
                    pygame.display.update()        

                #frame_ct += 1 
                
                
            else: #  <<<<<<<<<<<<<<<<<<<< 2nd viz !!!!!
                #print(imgs)
               
                if frame_id < delays: frame_data = log_data[0]
                elif frame_id >= len(log_data): frame_data = log_data[-1] # frame DELAYS 
                else: frame_data = log_data[frame_id-delays]
                frame_ct = frame_id
                pygame.display.set_caption(f'Time Remaining: {time_remaining:01d}')        
                
                pygame.display.update()
                #if not sound is None:
                #    if frame_ct == sound_frame: sound.play()

                reward = 0
                prev_reward = 0
                if frame_ct > 1 and frame_ct < len(score):
                    # Tentative change to synchronize the flash and picking up the object
                    if frame_ct > delays: prev_reward =  score[frame_ct-(delays-1)-1] - score[frame_ct-delays-1]
                    reward = score[frame_ct-(delays-1)] - score[frame_ct-delays]
                            
                if frame_ct-delays < len(log_data):
                    self.render_frame(frame_data, direction[frame_ct-delays], reward)       
                else:
                    self.render_frame(frame_data, direction[-1], reward)             
                if len(imgs) > 0 : #and frame_ct < 400
                    if frame_ct in imgs: face_frames = imgs[frame_ct]    
                    else: continue
                    num_frames = len(face_frames)
                    #face_frames.sort()    
                    face_frame_ct = 0
                    x0, y0 = self.last_head  
                    imm_coords = pygame.Rect(
                        x0*self.CELL_SIZE,
                        y0*self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    ) 
                    x, y = self.curr_head                    
                    
                    for face_frame in face_frames:   
                        # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, imm_coords)
                        self.fill_background(imm_coords)
                        if reward != 0:
                            imm_coords = pygame.Rect(
                                        x0*self.CELL_SIZE + (x-x0)*min(face_frame_ct,(num_frames-2))*self.CELL_SIZE//num_frames,
                                        y0*self.CELL_SIZE + (y-y0)*min(face_frame_ct,(num_frames-2))*self.CELL_SIZE//num_frames,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE,
                            )
                            radius =  self.CELL_SIZE // 2 
                            cell_center = (x0*self.CELL_SIZE + (x-x0)*min(face_frame_ct,(num_frames-2))*self.CELL_SIZE//num_frames + radius,
                                           y0*self.CELL_SIZE + (y-y0)*min(face_frame_ct,(num_frames-2))*self.CELL_SIZE//num_frames + radius)
                            
                        else:
                            imm_coords = pygame.Rect(
                                        x0*self.CELL_SIZE + (x-x0)*face_frame_ct*self.CELL_SIZE//num_frames,
                                        y0*self.CELL_SIZE + (y-y0)*face_frame_ct*self.CELL_SIZE//num_frames,
                                        self.CELL_SIZE,
                                        self.CELL_SIZE,
                            )
                            radius =  self.CELL_SIZE // 2 
                            cell_center = (x0*self.CELL_SIZE + (x-x0)*face_frame_ct*self.CELL_SIZE//num_frames + radius,
                                           y0*self.CELL_SIZE + (y-y0)*face_frame_ct*self.CELL_SIZE//num_frames + radius)
                            
                        # restore icon from last frame if crashed
                        cell_coords = pygame.Rect(
                            self.curr_head[0] * self.CELL_SIZE,
                            self.curr_head[1] * self.CELL_SIZE,
                            self.CELL_SIZE,
                            self.CELL_SIZE,
                        )

                        flash_finished = False  
                        if face_frame_ct < num_frames//2:
                            if reward == self.object_type_to_reward[CellType.GOOD_FRUIT]: # good fruit
                                color = Colors.CELL_TYPE[CellType.GOOD_FRUIT]
                                internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                                pygame.draw.rect(self.screen, color, internal_square_coords)
                            elif reward == self.object_type_to_reward[CellType.BAD_FRUIT]:
                                color = Colors.CELL_TYPE[CellType.BAD_FRUIT]
                                internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                                pygame.draw.rect(self.screen, color, internal_square_coords)
                            elif reward == self.object_type_to_reward[CellType.LAVA]:
                                color = Colors.CELL_TYPE[CellType.LAVA]
                                internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                                pygame.draw.rect(self.screen, color, internal_square_coords)
                        else:
                            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
                            self.fill_background(cell_coords)
                            if reward != 0 and reward is not None and self.flash_background:
                                self.faded_flash(prev_reward, num_frames)
                                flash_finished = True

                        if self.flash_background and not flash_finished: 
                            self.faded_flash(prev_reward, num_frames)

                                        
                        ''' update head'''
                        pygame.draw.circle(self.screen, Colors.CELL_TYPE[CellType.ROBOTAXI_HEAD], cell_center, radius*2//3)       
                                
                        ''' Face recordings'''                 
                        face_img = pygame.transform.scale(pygame.image.load(face_frame),(320,240))
                        self.screen.blit(face_img, pygame.Rect(
                                self.screen_size[0] - 5*self.CELL_SIZE , 
                                20,
                                320,
                                240))
                        large_face = detect_face(cv2.imread(face_frame,cv2.IMREAD_COLOR))                        
                        if not large_face is None:
                            lface_img = pygame.transform.scale(pygame.surfarray.make_surface(large_face),(320,320))
                            self.screen.blit(pygame.transform.flip(pygame.transform.rotate(lface_img,-90),1,0), 
                                pygame.Rect(
                                        self.screen_size[0] - 5*self.CELL_SIZE, 270, 320, 320))                        
                        
                        ''' Time left panel '''
                        ct = 3
                        text = ("Time Left",str(int(round(time_remaining/(self.FPS_LIMIT/1.5)))))
                        disp_text = self.text_font.render(text[0], True, (0, 0, 0))
                        if time_remaining < self.time_thresh and time_remaining%2 == 0:
                            disp_num = self.num_font.render(text[1], True, (225, 50, 50))
                        else:
                            disp_num = self.num_font.render(text[1], True, (50, 205, 50))
                        self.screen.blit(disp_text, (self.screen_size[0] - 4.75*self.CELL_SIZE , 35+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct + 60 - disp_text.get_height() ))
                        ct += 1
                        cell_coords = pygame.Rect(
                            self.screen_size[0] - 4.5*self.CELL_SIZE ,
                            120+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct + 10 - disp_num.get_height(),
                            2.5*self.CELL_SIZE,
                            disp_num.get_height()-10,
                        )
                        pygame.draw.rect(self.screen, (59,59,59), cell_coords)
                        self.screen.blit(disp_num, (self.screen_size[0] - 4.1*self.CELL_SIZE , 65+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct + 60 - disp_num.get_height() ))
                        if frame_ct > 0 and self.save_frames:
                            pygame.image.save(self.screen, 'screenshots/replay_frame_%s_%05d.jpg' % (exp_id, face_img_total_ct))
                            face_img_total_ct += 1
                        
                        pygame.display.update()
                        self.fps_clock.tick(self.FPS_LIMIT*(num_frames)*10) 
                        
                        face_frame_ct += 1
                                        

                else: # No face imgs
                    
                    x0, y0 = self.last_head  
                    imm_coords = pygame.Rect(
                        x0*self.CELL_SIZE,
                        y0*self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    ) 
                    x, y = self.curr_head 

                    for interpolate_idx in range(1,self.intermediate_frames-1):
                        self.fill_background(imm_coords)
                        
                        imm_coords = pygame.Rect(
                                    x0*self.CELL_SIZE + (x-x0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames,
                                    y0*self.CELL_SIZE + (y-y0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames,
                                    self.CELL_SIZE,
                                    self.CELL_SIZE,
                        )
                        radius =  self.CELL_SIZE // 2 
                        cell_center = (x0*self.CELL_SIZE + (x-x0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames + radius,
                                       y0*self.CELL_SIZE + (y-y0)*interpolate_idx*self.CELL_SIZE//self.intermediate_frames + radius)
                        
                        
                        # restore icon from last frame if crashed
                        cell_coords = pygame.Rect(
                            self.curr_head[0] * self.CELL_SIZE,
                            self.curr_head[1] * self.CELL_SIZE,
                            self.CELL_SIZE,
                            self.CELL_SIZE,
                        )

                        flash_finished = False
                        if interpolate_idx < self.intermediate_frames//2:
                            if reward == self.object_type_to_reward[CellType.GOOD_FRUIT]: # good fruit
                                color = Colors.CELL_TYPE[CellType.GOOD_FRUIT]
                                internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                                pygame.draw.rect(self.screen, color, internal_square_coords)
                            elif reward == self.object_type_to_reward[CellType.BAD_FRUIT]:
                                color = Colors.CELL_TYPE[CellType.BAD_FRUIT]
                                internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                                pygame.draw.rect(self.screen, color, internal_square_coords)
                            elif reward == self.object_type_to_reward[CellType.LAVA]:
                                color = Colors.CELL_TYPE[CellType.LAVA]
                                internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
                                pygame.draw.rect(self.screen, color, internal_square_coords)
                        else:
                            self.fill_background(cell_coords)
                            if reward != 0 and reward is not None and self.flash_background:
                                self.faded_flash(reward, self.intermediate_frames)
                                flash_finished = True

                        if self.flash_background and not flash_finished:
                            self.faded_flash(reward, self.intermediate_frames)
                                        
                        ''' update head'''
                        pygame.draw.circle(self.screen, Colors.CELL_TYPE[CellType.ROBOTAXI_HEAD], cell_center, radius*2//3)

                        ''' Time left panel '''                         
                        ct = 2
                        text = ("Time Left",str(max(0,int(round(time_remaining/(self.FPS_LIMIT/1.5))))))
                        disp_text = self.text_font.render(text[0], True, (0, 0, 0))
                        if time_remaining < self.time_thresh and time_remaining%2 == 0:
                            disp_num = self.num_font.render(text[1], True, (225, 50, 50))
                        else:
                            disp_num = self.num_font.render(text[1], True, (50, 205, 50))
                        self.screen.blit(disp_text, (self.screen_size[0] - 4.75*self.CELL_SIZE , 35+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct + 60 - disp_text.get_height() ))
                        ct += 1
                        cell_coords = pygame.Rect(
                            self.screen_size[0] - 4.5*self.CELL_SIZE ,
                            120+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct + 10 - disp_num.get_height(),
                            2.5*self.CELL_SIZE,
                            disp_num.get_height()-10,
                        )
                        pygame.draw.rect(self.screen, (59,59,59), cell_coords)
                        self.screen.blit(disp_num, (self.screen_size[0] - 4.1*self.CELL_SIZE , 65+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct + 60 - disp_num.get_height() ))

                        #if self.save_frames:
                        #    pygame.image.save(self.screen, 'screenshots/replay_frame_%s_%03d.jpg' % (exp_id, frame_ct))
                        pygame.display.update()
                        # self.fps_clock.tick(self.FPS_LIMIT)
                        self.fps_clock.tick(self.FPS_LIMIT*self.intermediate_frames)
                 

def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = argparse.ArgumentParser(
        description='Replay recorded 20*20 robotaxi game from log.'
    )
    
    parser.add_argument(
        '--cond', 
        required=True,
        type=int, 
        help='user study condition.'
    )

    parser.add_argument(
        '--log_file', 
        type=str, 
        help='log file name.'
    )

    parser.add_argument(
        '--original', 
        action="store_true", 
        default=True, 
        help='render original visualization.'
    )
    
    parser.add_argument(
        '--save_frames', 
        action="store_true", 
        default=False, 
        help='save frames as jpg files in screenshots/ folder.'
    )
    
    parser.add_argument(
        '--num_noise_color',  
        default=0, 
        type=int, 
        help='number of noise features.'
    )

    parser.add_argument(
        '--noise_prob',  
        default=0.05, 
        type=float, 
        help='probability of a cell being noise.'
    )
    
    parser.add_argument(
        '--capture',  
        action="store_true", 
        default=True, 
        help='capture webcam images and audio input.'
    )
    
    parser.add_argument(
        '--participant', 
        type=str, 
        default='test',
        help='participant id'
    )
    
   
    
    return parser.parse_args(args)


def main():
    
    if not os.path.exists('./log/'): os.makedirs('./log/') 
    if not os.path.exists('./csv/'): os.makedirs('./csv/') 

     # pre selected conditions for user study
    '''
    data_files = {0:['recordings/user_study_logs2/episode_30.record'],
                  1:['recordings/user_study_logs2/episode_32.record',
                     'recordings/user_study_logs2/episode_36.record',
                     'recordings/user_study_logs2/episode_49.record'],
                  2:['recordings/user_study_logs2/episode_5.record',
                     'recordings/user_study_logs/episode_34.record'],
                  3:['recordings/autocar_16_static.log']   }
    '''              
    data_files = [
                  'recordings/new_user_study/episode_2.record',  # -3
                  'recordings/new_user_study/episode_37.record', # -13
                  'recordings/new_user_study/episode_41.record', # 22
                  'recordings/new_user_study/episode_31.record', # 7
                  'recordings/new_user_study/episode_30.record', # -25
                  'recordings/new_user_study/episode_46.record', # 24
                  ] 
                  
                  
    parsed_args = parse_command_line_args(sys.argv[1:])
    stationary = False
    #if parsed_args.cond == len(data_files) - 1: stationary = True
    
    if stationary: PyGameGUI.FPS_LIMIT = 2.0
    else: PyGameGUI.FPS_LIMIT = 1.2
    log_data = []
    line_ct = 0
    frame_data = []
    hit_wall = []
    direction = []
    total_reward = 0
    score = [0]
    max_step_limit = 0
    hit_wall_poses = []
    agent = []
    if stationary: field_size = 16
    else: field_size = 8
    
    #log_file_handle = open(parsed_args.log_file)

    logfile_name = data_files[parsed_args.cond] #random.choice()
    log_file_handle = open(logfile_name)

    for line in log_file_handle:
        line = line.strip()
        if line.startswith('6'):
            frame_data.append(line)
            line_ct += 1

        if line.startswith('punch'):
            hit_wall.append(eval(line.split(':')[1]))

        if line.startswith('pwall_pos'):
            hit_wall_poses.append(eval(line.split(':')[1]))
            
        if line.startswith('direction'):
            x = int(line.split('(', 1)[1].split(')')[0].split(',')[0])
            y = int(line.split('(', 1)[1].split(')')[0].split(',')[1])
            direction.append(Point(x,y))

        if line.startswith('R'):
            total_reward = total_reward + eval(line.split('=')[1])
            score.append(total_reward)
        
        if line.startswith('Agent'):
            agent.append(int(line.split(':')[1]))
     
        if line.startswith('max_step_limit'):
            max_step_limit = eval(line.split(':')[1])

        if line_ct == field_size:
            line_ct = 0
            log_data.append(list(frame_data))
            frame_data = []

    hit_wall.append(False)
    game_replay = PyGameGUI(field_size=field_size, original=parsed_args.original, num_noise_color=parsed_args.num_noise_color, noise_prob=parsed_args.noise_prob, save_frames=parsed_args.save_frames, proxy=0, stationary=stationary, agent=agent)
    
    try:        
        exp_id = logfile_name.split('/')[-1].split('.')[0]
        if parsed_args.capture:
            data_dir = 'user_study_data/'
            
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)   
            if not os.path.exists(data_dir+'audio_log/'):
                os.makedirs(data_dir+'audio_log/')
            if not os.path.exists(data_dir+'webcam_imgs/'):
                os.makedirs(data_dir+'webcam_imgs/')
            if not os.path.exists(data_dir+'webcam_imgs/'+parsed_args.participant):
                os.makedirs(data_dir+'webcam_imgs/'+parsed_args.participant)
            if not os.path.exists(data_dir+'webcam_imgs/'+parsed_args.participant+'/'+exp_id):
                os.makedirs(data_dir+'webcam_imgs/'+parsed_args.participant+'/'+exp_id)
                
            capture_thread0 = audioThread(0, "aud", 0, data_dir, parsed_args.participant, parsed_args.participant+'_'+exp_id)
            capture_thread1 = captureThread(1, "cam", 1,  data_dir, parsed_args.participant, exp_id, parsed_args.participant+'_'+exp_id)
            capture_thread0.start()
            capture_thread1.start()
        game_replay.render(log_data, hit_wall, hit_wall_poses, direction, score, max_step_limit, exp_id, participant=parsed_args.participant)
        print('Final Score:', score[-1])
        
        if parsed_args.capture:
            capture_thread0.stop()
            capture_thread1.stop()
            capture_thread0.join()
            capture_thread1.join()
    except QuitRequestedError:
        if parsed_args.capture:
            capture_thread0.stop()
            capture_thread1.stop()
            capture_thread0.join()
            capture_thread1.join()


if __name__ == '__main__':
    main()

