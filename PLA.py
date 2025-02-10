#Author: Marlon Dominguez
#Machine Learning/Rutgers University 

import matplotlib.pyplot as plt

print("PLA Algorithm")

Label = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1]
x1 = [5, 3, 5, 2, 3, 4, 5, 7, 8, 9, 7, 10, 10, 9]
x2 = [6, 4, 5, 5, 2, 6, 3, 8, 8, 8, 9, 9, 10, 10]

bias = 0; 

x1_weight = 0; 
x2_weight = 0; 

learning_rate = 0.2
#Boolean to control PLA loop count
Successful = False

while Successful == False:
    
    #Boolean to identify when we have achievied success
    #In other words, when we found the perfect line/boundary
    Unsuccessful = False
    
    for i in range(0, 14):
        
        #   The prediction is defined by the weights applied to preceptron 
        
        pred = (x1[i] * x1_weight) + (x2[i] * x2_weight) + bias
        
        if pred * Label[i] <= 0: #Line is not perfect, update weights
            
            bias += learning_rate * Label[i]
           
            x1_weight += (learning_rate * Label[i] * x1[i]) 
            x2_weight += (learning_rate * Label[i] * x2[i])
            
            Unsuccessful = True
            
    
    if Unsuccessful == False: #Line is perfect
        
        Successful = True
    
print()

print("Final x1 weight:", x1_weight)
print("Final x2 weight:", x2_weight)
print("Final bias:", bias)
print()

#Let's make a prediction for x1 = 3 and x2 = 4, which are the second values in x1 and x2, to see if it is correct.
pred = (x1[2] * x1_weight) + (x2[3] * x2_weight) + bias

print("Prediction for x1 = 3 and x2 = 4 is: ", pred)

print("Since the value is greater than 0, the prediction is Yes!")

# Plotting the data points (Class 1 vs Class -1)
plt.scatter(x1[:7], x2[:7], color='blue', label='Class 1 (yes)')
plt.scatter(x1[7:], x2[7:], color='red', label='Class -1 (no)')


# The decision boundary is w1 * x1 + w2 * x2 + b = 0
# Then x2 is x2 = (-w1 * x1 - b) / w2

#array of x1 values for drawing seperating line 
x1_range = [min(x1)-1, max(x1)+1]  # Range for x1 values

#Find x2 with decision boundary equation
x2_range = [(-x1_weight * x1_val - bias) / x2_weight for x1_val in x1_range]

#Show seperating line 
plt.plot(x1_range, x2_range, color='green', label='Decision Boundary')

#Title + Labels
plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Perceptron Decision Boundary')
plt.legend()


plt.show()