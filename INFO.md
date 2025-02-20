# CIFAR 10 

CIFAR 10 is dataset used commonly for training machine learning models. It consists of 60,000 images divided into 10 classes. 

1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

The images are of 32x32 pixels. Link: https://www.cs.toronto.edu/~kriz/cifar.html
<br>
<br>

# TEST CASES

For all the test cases, Cross-Entropy Loss function is used as if it is commonly used for multi-class classification. The optimizers used are SGD (with momentum) and Adam. Since we are doing standard classification tasks, cross-entropy loss + Adam/SGD is used.

Test Cases:
    -- Only Fully connected layer is changed to match CIFAR10 (all other layers are frozen)
        -- SGD + Momentum
            -- With L2 Regularization
            -- Without L2 Regularization
        -- Adam
            -- With L2 Regularization
            -- Without L2 Regularization
    -- Fully Connected layer and some final layers are changed (Rest all layers are frozen)
        -- SGD + Momentum
            -- With L2 Regularization
            -- Without L2 Regularization
        -- Adam
            -- With L2 Regularization
            -- Without L2 Regularization