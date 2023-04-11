import cv2
import numpy as np

blank = np.zeros((500, 500, 3))

start_point = (5, 5)
  

end_point = (220, 220)
  
color = (0, 0, 255)
  
thickness = cv2.FILLED
  
cv2.rectangle(blank, start_point, end_point, color, thickness)

## Same you draw circle and line
 
cv2.putText(blank, "Guddu", (0,100), cv2.FONT_HERSHEY_TRIPLEX, 4.5, (255, 255, 255), 2)

cv2.imshow("plane", blank)
cv2.waitKey(0)