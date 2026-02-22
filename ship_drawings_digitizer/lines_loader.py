import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import plotly.graph_objs as go
import plotly.offline as pyo

#TESTED WITH METERS AS UNITS

class ShipLines():
    """
    Defines a ShipLines object that will be able to parse json files and extract frame+profile data to a NAPA readable file
    """
    
    def __init__(self, particulars, output_file):
        self.use_theoretical_frames = True #this will use longitudinal values as-is, considered theoretical frames
                                           #will also add # symbol before frame number
        self.output_file = output_file
        self.LOA = particulars["LOA"]
        self.LBP = particulars["LBP"]
        self.B = particulars["B"]
        self.D = particulars["D"]
        self.T = particulars["T"]
        self.NumberFrames = particulars["Frames"] #Only given when frames are from A.P to F.P
        self.FrameSpacing = particulars["Spacing"] #Theoretical Frame Spacing
        self.max_y = self.B/2 #Max y: Avoid exceeding B/2
        self.max_z = round(self.D*1.3) #Max z: Add one extra point
        self.FRAMES={}
        self.PROFILE=[]
        self.accuracy = 2
        
        if self.NumberFrames!=0:
            self.fr_spacing = self.LBP/self.NumberFrames
            self.use_theoretical_frames = False #use classic 0-10 frames
        self.clear_output() #clear previous entries
        self.locate_midship()

    def simplifyPointCurve(self, pointArray):
        """
        Locate obsolete points and delete them
        Obsolete points are defined as the ones that exist on the line defined by the ones before and after them
        """
        newPoints = [pointArray[0]]
        for i in range(1,len(pointArray)-1):
            x1 = pointArray[i-1][0]
            y1 = pointArray[i-1][1]
            x2 = pointArray[i+1][0]
            y2 = pointArray[i+1][1]
            point_x = pointArray[i][0]
            point_y = pointArray[i][1]
            ignore = False
            try:
                linterp_y = (((y2-y1)/(x2-x1))*(point_x-x1)+y1)
                if linterp_y == point_y: ignore = True
            except ZeroDivisionError:
                linterp_x = (((x2-x1)/(y2-y1))*(point_y-y1)+x1)
                if linterp_x == point_x: ignore = True
            if not ignore:
                newPoints.append(pointArray[i])
        newPoints.append(pointArray[len(pointArray)-1])
        return newPoints
            
    def clear_output(self):
        """
        Clears the output file
        """
        with open(self.output_file, "a") as file: 
            file.seek(0, 0)
            file.truncate()
        print("Output file cleared")
        
    def locate_midship(self):
        """
        Locates approx frame number of midship
        """
        if self.use_theoretical_frames:
            self.midship = round((self.LBP/2)/self.FrameSpacing, self.accuracy)
        else:
            self.midship = round(self.LBP/2, self.accuracy)
            
            
    def load_JSON_points(self, jsonfile):
        with open(jsonfile, "r") as f: #load point dataset
            data = json.load(f)
            
        for dataset in range(len(data["datasetColl"])):
            linedata = data["datasetColl"][dataset]
            print(f'Loading line: {data["datasetColl"][dataset]["name"]}')
            
            if linedata["name"][:2]=="fr":
                print(f"Line {dataset} is a frame ({linedata['name']})")
                self.loadFrame(linedata, maxZ = self.max_z, digits=self.accuracy)

            elif linedata["name"]=="profile":
                print(f"Line {dataset} is the profile")
                self.loadProfile(linedata, digits=self.accuracy)
        
    def loadFrame(self, linedata, maxZ=0, digits=3, make_positive=True):
        """
        Ship Frame Loading function
        IN
        ---
        output: the file where you export the frame for NAPA
        linedata: a WebPlotDigitizer json dataset (calibrated points)
        maxZ: adding one more point for the same y as the last and a max z
        digits: rounding
        make_positive: fix ship lines (usually aft is shown starboard)

        OUT
        ---
        an additional element to loaded_frames dict
        writes output to file
        """
        make_straight = ""
        frame_decleration = f"cur, {linedata['name']}\n"
        if self.use_theoretical_frames: 
            fr_x = float(linedata['name'][2:])
            frame_x_position = f"x, #{fr_x}\n"
        else: 
            fr_x = round(float(linedata['name'][2:])*self.fr_spacing, digits) #Real X coord of frame
            frame_x_position = f"x, {fr_x}\n"

        #experimental - always add a point near midship on digitizing the profile curves!
        profile_connection = "stern"
        if fr_x >= self.midship:
            profile_connection = "stem"
        
        txt_line = ["yz", "*", make_straight, profile_connection, "/- -/"] #keep point order, make straight lines, connect to stern
        frame_array = []
        for point in linedata["data"]:
            y = round(float(point["value"][0]), digits)
            z = round(float(point["value"][1]), digits)
            if abs(y)>self.max_y: y=(abs(y)/y)*self.max_y
            if make_positive:
                y=abs(y)
                z=abs(z)
            frame_array.append((y,z))
        
        frame_array = self.simplifyPointCurve(frame_array)

        frame_points = [f"{str(point).replace(' ','')}," for point in frame_array]

        frame_line = " ".join(txt_line + frame_points)[:-1]

        if maxZ!=0:
            frame_array.append((y,maxZ)) #add a point with same y but maxZ
            frame_line = frame_line +  f" /- -/ {str((y,maxZ)).replace(' ','')},"

        #napa file new line segment (frame)
        frame_data = [
            frame_decleration,
            frame_x_position,
            frame_line,
            "\nok\n"]
        
        with open(self.output_file, "a") as file:
            file.writelines(frame_data)
        self.FRAMES[(linedata['name'], fr_x)] = frame_array

        
    
    def loadProfile(self, linedata, digits=3):
        
        stern_decleration = "cur, stern\n"
        stem_decleration = "cur, stem\n"
        profile_y = "y, 0\n"

        txt_line = ["xz", "*"]
        profile_array = []
        for point in linedata["data"]:
            if self.use_theoretical_frames:
                x = round(float(point["value"][0]), digits)
            else:
                x = round(float(point["value"][0])*self.fr_spacing, digits) #real_x of profile
                
            z = round(float(point["value"][1]), digits)
            z=abs(z)
            profile_array.append((x,z))
        self.PROFILE = profile_array
        #Implementation to divide profile into stern/stem
        #Needs a point to be near center (selects automatically nearest)
        long_center = self.midship
        profile_x = np.array([i[0] for i in profile_array])
        center_offset = (profile_x-long_center)
        break_point = np.argmin(abs(center_offset))
        print(f"Break point for end of stern/start of stem: {break_point}")
        toggle_hashtag = "("
        if self.use_theoretical_frames:
            toggle_hashtag = "(#"
        stern_points = [f"{str(point).replace(' ','').replace('(', toggle_hashtag)}," for point in profile_array[:break_point+1]]
        stem_points = [f"{str(point).replace(' ','').replace('(', toggle_hashtag)}," for point in profile_array[break_point:]]

        stern_line = " ".join(txt_line + stern_points)[:-1]
        stem_line = " ".join(txt_line + stem_points)[:-1]
        
        #napa file new line segment (stern & stem)
        profile_data = [
            stern_decleration,
            profile_y,
            stern_line,
            "\nok\n",
            stem_decleration,
            profile_y,
            stem_line,
            "\nok\n"
            ]

        with open(self.output_file, "a") as file:
            file.writelines(profile_data)




    def plot_lines(self):
        line_data = []
        
        #fig = plt.figure(figsize=(10, 8), dpi=500)
        #ax = plt.axes(projection='3d')

        #plotting the profile
        xPROFILE = np.array([i[0] for i in self.PROFILE])
        zPROFILE = np.array([i[1] for i in self.PROFILE])
        yPROFILE = np.zeros(len(xPROFILE))
        #ax.plot3D(xPROFILE, yPROFILE, zPROFILE)
        trace = go.Scatter3d(
            x=xPROFILE,
            y=yPROFILE,
            z=zPROFILE,
            mode='lines',
            name='profile')
        line_data.append(trace)
        #plotting the frames
        for frame in self.FRAMES:
            yFRAME = np.array([i[0] for i in self.FRAMES[frame]])
            zFRAME = np.array([i[1] for i in self.FRAMES[frame]])
            xFRAME = frame[1]*np.ones(len(yFRAME))

            trace = go.Scatter3d(
                x=xFRAME,
                y=yFRAME,
                z=zFRAME,
                mode='lines',
                name=f'{frame[0]}')

            line_data.append(trace)
            #ax.plot3D(xFRAME, yFRAME, zFRAME)
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X-Axis'),
                yaxis=dict(title='Y-Axis'),
                zaxis=dict(title='Z-Axis'),
                aspectmode='manual',  # Set the aspect mode to manual
                aspectratio=dict(x=1, y=self.B/self.LOA, z=self.max_z/self.LOA)
            ),
            title='3D Line Plot',
        )
        fig = go.Figure(data=line_data, layout=layout)
        pyo.plot(fig, filename='wowza_what_a_ship.html')
        #ax.set_box_aspect([1, self.B/self.LOA, self.max_z/self.LOA])
        
        #fig.savefig("test")
        
json_profile_data = "example_PROFILE.json" #json file from webplotdigitizer
json_frame_data = "example_FRAMES.json"
napa_file = "example_napa_lines.txt" #output file



particulars = {
    "LOA":352.25,
    "LBP":336.4,
    "B":42.8,
    "D":24.1,
    "T":15,
    "Frames":20,
    "Spacing":0}



"""
the json format is:
{
    "datasetColl":
    [
        {
            "name": "fr",
            "data":
            [
                {
                    "value": [x,y]
                },
                {
                    "value": [x,y]
                }
            ]
        }

    ]
}
"""

def frameDataToJSON(framesDict, filename):
    """
    Takes in a dictionary object of {string fr:DataFrame points.
    Outputs a json file similar to the one produced by webplot digitizer
    """
    jsonFile = {"datasetColl":[]}
    for frame in framesDict:
        newFrame = {}
        newFrame["name"] = frame
        data = []
        for point in range(len(framesDict[frame])):
            valuePair = [framesDict[frame].iloc[point][0],framesDict[frame].iloc[point][1]]
            data.append({"value":valuePair})
        newFrame["data"] = data
        jsonFile["datasetColl"].append(newFrame)
    output_file = 'data.json'

    # Export the data to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(jsonFile, json_file, indent=4)
        
        
frames={}
originalFile = pd.read_excel("OFFSETS.xlsx", header=None)
for column in range(0,len(originalFile.columns),5):
    frName = "fr"+str(originalFile[column].iloc[3])
    frData = originalFile[[column+2, column+3]].iloc[4:].dropna()  ##4: gets 2nd data point and on
    frData.columns = ["y", "z"]
    frames[frName] = frData
    frData.to_csv("csvFrames/"+frName+".csv", index=False)
frameDataToJSON(frames, "example_FRAMES.json")


example = ShipLines(particulars, napa_file)
example.load_JSON_points(json_profile_data)
example.load_JSON_points(json_frame_data)
example.plot_lines()
#################ADDING 3D XYZ LINES - NOT GENERALIZED#######################
#make extra lines
"""
ch1 = []
digits=2
loaded_frames = example.FRAMES
for frame in loaded_frames:
    fr_x = frame[1]
    y_points = [round(float(pair[0]), digits) for pair in loaded_frames[frame][1:4]] #1-4: points 2nd, 3rd and 4th (1,2,3)
    z_points = [round(float(pair[1]), digits) for pair in loaded_frames[frame][1:4]]
    ch1.append([fr_x, y_points[0], z_points[0]])


    
#################ADDING 3D XYZ LINES - NOT GENERALIZED#######################
#to generalize: parse a frame:index of point dict to a loaded_frames dataset which will extract the appropriate points

def trial(line):
    datapoints = [f"{str(i).replace('[','(#').replace(']',')').replace(' ','')}," for i in line]
    return datapoints
ch1a = " ".join(trial(ch1)[0:9])[:-1]
ch1f = " ".join(trial(ch1)[8:])[:-1]


newlines = [
    "cur, ch1a\n",
    "xyz * "+ch1a,
    "\nok\n",

    "cur, ch1f\n",
    "xyz * "+ch1f,
    "\nok\n"]

with open(napa_file, "a") as file:
    file.writelines(newlines)
"""
