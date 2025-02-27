# class to count the vehicles passing through a virtual line
class VirtualLineCounter:
    def __init__(self, line_y):
        self.line_y=line_y #horizontal line positioning
        self.counts={'north': 0, 'south': 0} #counts of vehicles crossing the line
        self.track_history={}

    def update(self, tracks):
        for track in tracks:
            x_center, y_center=track['position']
            if track['id'] not in self.track_history:
                self.track_history[track['id']]=[]

                #checking if the line is crossed?
                if len(self.track_history[track['id']])>0:
                    prev_y=self.track_history[track['id']][-1]

                    if (prev_y<self.line_y) and (y_center>=self.line_y):
                        self.counts['south']+=1
                    
                    elif(prev_y>self.line_y) and (y_center<=self.line_y):
                        self.counts['north']+=1
                    
                self.track_history[track['id']].append(y_center)
