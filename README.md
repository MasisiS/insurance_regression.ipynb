{
  "metadata": {
    "kernelspec": {
      "name": "python",
      "display_name": "Python (Pyodide)",
      "language": "python"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "python",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8"
    },
    "colab": {
      "provenance": []
    },
    "vscode": {
      "interpreter": {
        "hash": "909775f4c1c1337c10dd26386ad7fdd8e68c24cfe611dcacc1a198b9e09c3e1f"
      }
    }
  },
  "nbformat_minor": 4,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "source": "## Simple Linear Regression\n\nThe objective of this notebook is to show you how to apply the algorithm you learnt about in the main task. To start, we will import packages that will help us manipulate data and leverage the easy-to-use machine learning tools from the scikit-learn library. Then, we will load our data into a pandas DataFrame. ",
      "metadata": {
        "id": "4g4GIdGykuJu"
      }
    },
    {
      "cell_type": "code",
      "source": "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom sklearn.linear_model import LinearRegression\n%matplotlib inline\n",
      "metadata": {
        "id": "4OwSIQ_qLsXA",
        "trusted": true
      },
      "outputs": [],
      "execution_count": 2
    },
    {
      "cell_type": "code",
      "source": "# Import the data\ninsurance_charges = pd.read_csv(\"insurance.csv\")\ninsurance_charges.head()",
      "metadata": {
        "id": "z3Z4dQBomziZ",
        "outputId": "fc6fa430-092f-4b5d-eb7b-6da7e4a8d0be",
        "trusted": true
      },
      "outputs": [
        {
          "execution_count": 3,
          "output_type": "execute_result",
          "data": {
            "text/plain": "   age     sex     bmi  children smoker     region      charges\n0   19  female  27.900         0    yes  southwest  16884.92400\n1   18    male  33.770         1     no  southeast   1725.55230\n2   28    male  33.000         3     no  southeast   4449.46200\n3   33    male  22.705         0     no  northwest  21984.47061\n4   32    male  28.880         0     no  northwest   3866.85520",
            "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>age</th>\n      <th>sex</th>\n      <th>bmi</th>\n      <th>children</th>\n      <th>smoker</th>\n      <th>region</th>\n      <th>charges</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>19</td>\n      <td>female</td>\n      <td>27.900</td>\n      <td>0</td>\n      <td>yes</td>\n      <td>southwest</td>\n      <td>16884.92400</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>18</td>\n      <td>male</td>\n      <td>33.770</td>\n      <td>1</td>\n      <td>no</td>\n      <td>southeast</td>\n      <td>1725.55230</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>28</td>\n      <td>male</td>\n      <td>33.000</td>\n      <td>3</td>\n      <td>no</td>\n      <td>southeast</td>\n      <td>4449.46200</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>33</td>\n      <td>male</td>\n      <td>22.705</td>\n      <td>0</td>\n      <td>no</td>\n      <td>northwest</td>\n      <td>21984.47061</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>32</td>\n      <td>male</td>\n      <td>28.880</td>\n      <td>0</td>\n      <td>no</td>\n      <td>northwest</td>\n      <td>3866.85520</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
          },
          "metadata": {}
        }
      ],
      "execution_count": 3
    },
    {
      "cell_type": "markdown",
      "source": "A linear model describes a linear relationship between an input and output variable. We therefore need to make sure that the problem we are looking to model with Linear Regression indeed has a continuous input and output variable with a linear relationship. \n\nBelow we have a scatter plot of the salary data. We can observe that the more years of experience an employee has, the higher their salary. \nThis is a linear relationship that can be estimated with a simple linear relationship. The independent variable (years of experience) is on the x-axis and the dependent variable (salary) is on the y-axis.",
      "metadata": {
        "id": "wcLw-FN4Fz6Y"
      }
    },
    {
      "cell_type": "code",
      "source": "# The iloc function allows use to select rows from the dataframe for plotting.  \nx = insurance_charges.iloc[:,:1].values\ny = insurance_charges.iloc[:,-1].values\n\nplt.scatter(x,y,color = 'b')\nplt.title('Insurance: Scatter Plot of Age vs. Charges')\nplt.xlabel('age')\nplt.ylabel('charges')\nplt.show()",
      "metadata": {
        "trusted": true
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": "For comparison, our problem might have no continuous input variables, or the impact of the input on the output variable may not be linear, as in this example:",
      "metadata": {
        "id": "L0Jl9foLMFlj"
      }
    },
    {
      "cell_type": "markdown",
      "source": "Now that we have determined a linear relationship between salary and years of experience, we will fit a simple linear regression model using sklearn.",
      "metadata": {
        "id": "9BNV79xVMmWH"
      }
    },
    {
      "cell_type": "code",
      "source": "# Fit the model\ninsurance_charges = LinearRegression()\ninsurance_charges.fit(x,y)",
      "metadata": {
        "id": "EdXkF5MpkuKu",
        "outputId": "4ec61029-d713-415c-e327-885946e44c73"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
            ],
            "text/plain": [
              "LinearRegression()"
            ]
          },
          "execution_count": 5,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": "What does it mean that we have fitted a model? It means we can make predictions along a line by inputting any x value. We can see how similar those predictions are to the x values for which we know the corresponding y values. Below these predictions are shown in red, and the observed values are in blue. ",
      "metadata": {
        "id": "CcFtzhaXkuKy"
      }
    },
    {
      "cell_type": "code",
      "source": "# Plot the data and model\ny_pred = insurance_charges.predict(x)\n\nplt.scatter(x,y,color = 'b')\nplt.title('Insurance: Scatter Plot of Age vs. Charges')\nplt.plot(x,y_pred,color = 'r')\nplt.xlabel('age')\nplt.ylabel('charges')\nplt.show()",
      "metadata": {
        "trusted": true
      },
      "outputs": [],
      "execution_count": null
    },
    {
      "cell_type": "markdown",
      "source": "If we are happy with this model, we can use it to determine how much to pay an employee who has been at the company longer than any other previous employee. In other words, we will extrapolate the linear model to predict a salary outside of the known data.",
      "metadata": {
        "id": "nKjw8-pfM3rV"
      }
    },
    {
      "cell_type": "code",
      "source": "# Predict an unknown value\nunk_x = [[12]] \n\nx_pred = np.append(x, unk_x).reshape(-1,1)\ny_pred = insurance_charges.predict(x_pred)\n\nplt.scatter(x,y,color = 'b')\nplt.title('Insurance: Scatter Plot of Age vs. Charges (with line of best-fit)')\nplt.plot(x_pred,y_pred,color = 'r')\nplt.xlabel('age')\nplt.ylabel('charges')\nplt.show()\n\nprint(\"Insurance charge per age should be:\", insurance_charges.predict(unk_x))",
      "metadata": {
        "trusted": true
      },
      "outputs": [],
      "execution_count": null
    }
  ]
}
