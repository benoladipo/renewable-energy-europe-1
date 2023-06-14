{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "4pf0OvInF8CO"
   },
   "outputs": [],
   "source": [
    "import time\n",
    "from datetime import datetime\n",
    "import random\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sqlite3\n",
    "import requests\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from statsmodels.tools.eval_measures import rmse\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import tensorflow  as tf\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow import keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "s_A_M5_dH1DY"
   },
   "outputs": [],
   "source": [
    "df1= pd.read_csv(\"/content/ninja_pv_wind_profiles_singleindex.csv\")\n",
    "df2= pd.read_csv(\"/content/weather_data.csv\")\n",
    "df3= pd.read_csv(\"/content/Final Merged.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 505
    },
    "id": "7ObL7-FUIDlV",
    "outputId": "4492603b-93a2-4617-d27f-36e297edba37"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "  <div id=\"df-34142cc9-0155-46d1-8126-d1624772e812\">\n",
       "    <div class=\"colab-df-container\">\n",
       "      <div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>time</th>\n",
       "      <th>NL_pv_national_current</th>\n",
       "      <th>NL_wind_national_current</th>\n",
       "      <th>NL_wind_offshore_current</th>\n",
       "      <th>NL_wind_onshore_current</th>\n",
       "      <th>NL_wind_national_long-termfuture</th>\n",
       "      <th>NL_wind_national_near-termfuture</th>\n",
       "      <th>NL_wind_offshore_near-termfuture</th>\n",
       "      <th>NL_wind_onshore_near-termfuture</th>\n",
       "      <th>NL_temperature</th>\n",
       "      <th>NL_radiation_direct_horizontal</th>\n",
       "      <th>NL_radiation_diffuse_horizontal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1980-01-01T00:00:00Z</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.4789</td>\n",
       "      <td>0.6650</td>\n",
       "      <td>0.4465</td>\n",
       "      <td>0.6132</td>\n",
       "      <td>0.5416</td>\n",
       "      <td>0.5501</td>\n",
       "      <td>0.3748</td>\n",
       "      <td>2.382</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1980-01-01T01:00:00Z</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.4590</td>\n",
       "      <td>0.6215</td>\n",
       "      <td>0.4308</td>\n",
       "      <td>0.5659</td>\n",
       "      <td>0.5215</td>\n",
       "      <td>0.5295</td>\n",
       "      <td>0.3635</td>\n",
       "      <td>2.236</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1980-01-01T02:00:00Z</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.4307</td>\n",
       "      <td>0.5601</td>\n",
       "      <td>0.4081</td>\n",
       "      <td>0.5156</td>\n",
       "      <td>0.5081</td>\n",
       "      <td>0.5172</td>\n",
       "      <td>0.3302</td>\n",
       "      <td>2.086</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1980-01-01T03:00:00Z</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.3983</td>\n",
       "      <td>0.5143</td>\n",
       "      <td>0.3781</td>\n",
       "      <td>0.4803</td>\n",
       "      <td>0.5014</td>\n",
       "      <td>0.5113</td>\n",
       "      <td>0.3086</td>\n",
       "      <td>1.861</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1980-01-01T04:00:00Z</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.3643</td>\n",
       "      <td>0.4591</td>\n",
       "      <td>0.3478</td>\n",
       "      <td>0.4388</td>\n",
       "      <td>0.4792</td>\n",
       "      <td>0.4899</td>\n",
       "      <td>0.2712</td>\n",
       "      <td>1.713</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11531</th>\n",
       "      <td>1981-04-25T11:00:00Z</td>\n",
       "      <td>0.395</td>\n",
       "      <td>0.3212</td>\n",
       "      <td>0.4526</td>\n",
       "      <td>0.2984</td>\n",
       "      <td>0.3190</td>\n",
       "      <td>0.2745</td>\n",
       "      <td>0.2730</td>\n",
       "      <td>0.3032</td>\n",
       "      <td>10.176</td>\n",
       "      <td>49.6251</td>\n",
       "      <td>336.9838</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11532</th>\n",
       "      <td>1981-04-25T12:00:00Z</td>\n",
       "      <td>0.314</td>\n",
       "      <td>0.2606</td>\n",
       "      <td>0.3673</td>\n",
       "      <td>0.2420</td>\n",
       "      <td>0.2730</td>\n",
       "      <td>0.2706</td>\n",
       "      <td>0.2729</td>\n",
       "      <td>0.2258</td>\n",
       "      <td>10.768</td>\n",
       "      <td>33.1483</td>\n",
       "      <td>309.2344</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11533</th>\n",
       "      <td>1981-04-25T13:00:00Z</td>\n",
       "      <td>0.220</td>\n",
       "      <td>0.2158</td>\n",
       "      <td>0.3359</td>\n",
       "      <td>0.1949</td>\n",
       "      <td>0.2867</td>\n",
       "      <td>0.2905</td>\n",
       "      <td>0.2977</td>\n",
       "      <td>0.1479</td>\n",
       "      <td>10.793</td>\n",
       "      <td>14.1345</td>\n",
       "      <td>235.4456</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11534</th>\n",
       "      <td>1981-04-25T14:00:00Z</td>\n",
       "      <td>0.140</td>\n",
       "      <td>0.2327</td>\n",
       "      <td>0.3892</td>\n",
       "      <td>0.2055</td>\n",
       "      <td>0.3525</td>\n",
       "      <td>0.3432</td>\n",
       "      <td>0.3535</td>\n",
       "      <td>0.1409</td>\n",
       "      <td>10.503</td>\n",
       "      <td>6.6136</td>\n",
       "      <td>168.9883</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11535</th>\n",
       "      <td>1981-04-25T15:00:00Z</td>\n",
       "      <td>0.090</td>\n",
       "      <td>0.2837</td>\n",
       "      <td>0.4476</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>11536 rows × 12 columns</p>\n",
       "</div>\n",
       "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-34142cc9-0155-46d1-8126-d1624772e812')\"\n",
       "              title=\"Convert this dataframe to an interactive table.\"\n",
       "              style=\"display:none;\">\n",
       "        \n",
       "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
       "       width=\"24px\">\n",
       "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
       "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
       "  </svg>\n",
       "      </button>\n",
       "      \n",
       "  <style>\n",
       "    .colab-df-container {\n",
       "      display:flex;\n",
       "      flex-wrap:wrap;\n",
       "      gap: 12px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert {\n",
       "      background-color: #E8F0FE;\n",
       "      border: none;\n",
       "      border-radius: 50%;\n",
       "      cursor: pointer;\n",
       "      display: none;\n",
       "      fill: #1967D2;\n",
       "      height: 32px;\n",
       "      padding: 0 0 0 0;\n",
       "      width: 32px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert:hover {\n",
       "      background-color: #E2EBFA;\n",
       "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
       "      fill: #174EA6;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert {\n",
       "      background-color: #3B4455;\n",
       "      fill: #D2E3FC;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert:hover {\n",
       "      background-color: #434B5C;\n",
       "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
       "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
       "      fill: #FFFFFF;\n",
       "    }\n",
       "  </style>\n",
       "\n",
       "      <script>\n",
       "        const buttonEl =\n",
       "          document.querySelector('#df-34142cc9-0155-46d1-8126-d1624772e812 button.colab-df-convert');\n",
       "        buttonEl.style.display =\n",
       "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
       "\n",
       "        async function convertToInteractive(key) {\n",
       "          const element = document.querySelector('#df-34142cc9-0155-46d1-8126-d1624772e812');\n",
       "          const dataTable =\n",
       "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
       "                                                     [key], {});\n",
       "          if (!dataTable) return;\n",
       "\n",
       "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
       "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
       "            + ' to learn more about interactive tables.';\n",
       "          element.innerHTML = '';\n",
       "          dataTable['output_type'] = 'display_data';\n",
       "          await google.colab.output.renderOutput(dataTable, element);\n",
       "          const docLink = document.createElement('div');\n",
       "          docLink.innerHTML = docLinkHtml;\n",
       "          element.appendChild(docLink);\n",
       "        }\n",
       "      </script>\n",
       "    </div>\n",
       "  </div>\n",
       "  "
      ],
      "text/plain": [
       "                       time  NL_pv_national_current  NL_wind_national_current  \\\n",
       "0      1980-01-01T00:00:00Z                   0.000                    0.4789   \n",
       "1      1980-01-01T01:00:00Z                   0.000                    0.4590   \n",
       "2      1980-01-01T02:00:00Z                   0.000                    0.4307   \n",
       "3      1980-01-01T03:00:00Z                   0.000                    0.3983   \n",
       "4      1980-01-01T04:00:00Z                   0.000                    0.3643   \n",
       "...                     ...                     ...                       ...   \n",
       "11531  1981-04-25T11:00:00Z                   0.395                    0.3212   \n",
       "11532  1981-04-25T12:00:00Z                   0.314                    0.2606   \n",
       "11533  1981-04-25T13:00:00Z                   0.220                    0.2158   \n",
       "11534  1981-04-25T14:00:00Z                   0.140                    0.2327   \n",
       "11535  1981-04-25T15:00:00Z                   0.090                    0.2837   \n",
       "\n",
       "       NL_wind_offshore_current  NL_wind_onshore_current  \\\n",
       "0                        0.6650                   0.4465   \n",
       "1                        0.6215                   0.4308   \n",
       "2                        0.5601                   0.4081   \n",
       "3                        0.5143                   0.3781   \n",
       "4                        0.4591                   0.3478   \n",
       "...                         ...                      ...   \n",
       "11531                    0.4526                   0.2984   \n",
       "11532                    0.3673                   0.2420   \n",
       "11533                    0.3359                   0.1949   \n",
       "11534                    0.3892                   0.2055   \n",
       "11535                    0.4476                   0.0000   \n",
       "\n",
       "       NL_wind_national_long-termfuture  NL_wind_national_near-termfuture  \\\n",
       "0                                0.6132                            0.5416   \n",
       "1                                0.5659                            0.5215   \n",
       "2                                0.5156                            0.5081   \n",
       "3                                0.4803                            0.5014   \n",
       "4                                0.4388                            0.4792   \n",
       "...                                 ...                               ...   \n",
       "11531                            0.3190                            0.2745   \n",
       "11532                            0.2730                            0.2706   \n",
       "11533                            0.2867                            0.2905   \n",
       "11534                            0.3525                            0.3432   \n",
       "11535                               NaN                               NaN   \n",
       "\n",
       "       NL_wind_offshore_near-termfuture  NL_wind_onshore_near-termfuture  \\\n",
       "0                                0.5501                           0.3748   \n",
       "1                                0.5295                           0.3635   \n",
       "2                                0.5172                           0.3302   \n",
       "3                                0.5113                           0.3086   \n",
       "4                                0.4899                           0.2712   \n",
       "...                                 ...                              ...   \n",
       "11531                            0.2730                           0.3032   \n",
       "11532                            0.2729                           0.2258   \n",
       "11533                            0.2977                           0.1479   \n",
       "11534                            0.3535                           0.1409   \n",
       "11535                               NaN                              NaN   \n",
       "\n",
       "       NL_temperature  NL_radiation_direct_horizontal  \\\n",
       "0               2.382                          0.0000   \n",
       "1               2.236                          0.0000   \n",
       "2               2.086                          0.0000   \n",
       "3               1.861                          0.0000   \n",
       "4               1.713                          0.0000   \n",
       "...               ...                             ...   \n",
       "11531          10.176                         49.6251   \n",
       "11532          10.768                         33.1483   \n",
       "11533          10.793                         14.1345   \n",
       "11534          10.503                          6.6136   \n",
       "11535             NaN                             NaN   \n",
       "\n",
       "       NL_radiation_diffuse_horizontal  \n",
       "0                               0.0000  \n",
       "1                               0.0000  \n",
       "2                               0.0000  \n",
       "3                               0.0000  \n",
       "4                               0.0000  \n",
       "...                                ...  \n",
       "11531                         336.9838  \n",
       "11532                         309.2344  \n",
       "11533                         235.4456  \n",
       "11534                         168.9883  \n",
       "11535                              NaN  \n",
       "\n",
       "[11536 rows x 12 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "id": "dlDiAvIdK8hE"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X = df3.drop(['NL_wind_national_long-termfuture','time'],axis=1)\n",
    "y = df3['NL_wind_national_long-termfuture']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "mYri3_6IVALF",
    "outputId": "2b084fab-b32d-470f-c8e2-084ee710c984"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7378    0.5215\n",
       "5915    0.1512\n",
       "234     0.0993\n",
       "7869    0.6173\n",
       "5088    0.1276\n",
       "         ...  \n",
       "599     0.1303\n",
       "5695    0.0078\n",
       "8006    0.8631\n",
       "1361    0.1516\n",
       "1547    0.2509\n",
       "Name: NL_wind_national_long-termfuture, Length: 8075, dtype: float64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "v_T4QCIrUxVz",
    "outputId": "55693e2d-fd99-43c1-81f7-d72f8bd55a04"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8075,)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "\n",
    "y_train.shape\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "SAmY2BwXUxSQ"
   },
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "y_train = y_train.reshape((y_train.shape[0], 1, x_train.shape[1]))\n",
    "y_test = y_test.reshape((y_test.shape[0], 1, x_test.shape[1]))\n",
    "y_train.shape\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Ad-Ts8lgUxPn"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "vsrOBUgCUxNb"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "PyO42VKYUxLF"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "Haft3jVJUxIo"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "BIneUkQBUxGY"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "8kvPdToqUxDy"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "F4qQQcfTUw4s"
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 241
    },
    "id": "yNAx437WScYl",
    "outputId": "780ee127-4508-4e53-ad7e-f7441c11094c"
   },
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "ignored",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-16-34c189beaaf4>\u001b[0m in \u001b[0;36m<cell line: 4>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mnr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mseed\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m9922\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mlabels\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdf3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mindx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mFeatures\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mindx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mms\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_test_split\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindx\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_size\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0.25\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;31m# print(indx)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'Features' is not defined"
     ]
    }
   ],
   "source": [
    "import numpy.random as nr\n",
    "nr.seed(9922)\n",
    "labels = np.array(df3)\n",
    "indx = range(Features.shape[0])\n",
    "indx = ms.train_test_split(indx, test_size = 0.25)\n",
    "# print(indx)\n",
    "x_train = Features[indx[0],:]\n",
    "y_train = np.ravel(labels[indx[0]])\n",
    "x_test = Features[indx[1],:]\n",
    "y_test = np.ravel(labels[indx[1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 408
    },
    "id": "wFv7DLE8QIDe",
    "outputId": "3ee742ce-f68b-43aa-c693-40277e87f233"
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "ignored",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-23-f7086f2ff890>\u001b[0m in \u001b[0;36m<cell line: 5>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mscaler\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mscaled_train_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mscaler\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtransform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mscaled_test_data\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mscaler\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtransform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_train\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/utils/_set_output.py\u001b[0m in \u001b[0;36mwrapped\u001b[0;34m(self, X, *args, **kwargs)\u001b[0m\n\u001b[1;32m    138\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mwraps\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mf\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    139\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mwrapped\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 140\u001b[0;31m         \u001b[0mdata_to_wrap\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    141\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdata_to_wrap\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtuple\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    142\u001b[0m             \u001b[0;31m# only wrap the first output for cross decomposition\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_data.py\u001b[0m in \u001b[0;36mtransform\u001b[0;34m(self, X, copy)\u001b[0m\n\u001b[1;32m    990\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    991\u001b[0m         \u001b[0mcopy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcopy\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mcopy\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcopy\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 992\u001b[0;31m         X = self._validate_data(\n\u001b[0m\u001b[1;32m    993\u001b[0m             \u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    994\u001b[0m             \u001b[0mreset\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[0;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[1;32m    563\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Validation should be done on X, y or both.\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    564\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mno_val_X\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mno_val_y\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 565\u001b[0;31m             \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput_name\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"X\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    566\u001b[0m             \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    567\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mno_val_X\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mno_val_y\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)\u001b[0m\n\u001b[1;32m    900\u001b[0m             \u001b[0;31m# If input is 1D raise error\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    901\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mndim\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 902\u001b[0;31m                 raise ValueError(\n\u001b[0m\u001b[1;32m    903\u001b[0m                     \u001b[0;34m\"Expected 2D array, got 1D array instead:\\narray={}.\\n\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    904\u001b[0m                     \u001b[0;34m\"Reshape your data either using array.reshape(-1, 1) if \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: Expected 2D array, got 1D array instead:\narray=[8075.].\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "scaled_train_data = scaler.transform(X_train)\n",
    "scaled_test_data = scaler.transform(y_train.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 443
    },
    "id": "_FoVC1OEUReu",
    "outputId": "94311f2e-eea0-4065-c9d2-95ffbf61027f"
   },
   "outputs": [
    {
     "ename": "InvalidIndexError",
     "evalue": "ignored",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   3801\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3802\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mcasted_key\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3803\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyError\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/_libs/index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/_libs/index.pyx\u001b[0m in \u001b[0;36mpandas._libs.index.IndexEngine.get_loc\u001b[0;34m()\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: '(slice(None, None, None), slice(10, None, None))' is an invalid key",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[0;31mInvalidIndexError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-25-d379a333e0dc>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpreprocessing\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mscaler\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpreprocessing\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mStandardScaler\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mx_train\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mscaler\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtransform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0my_test\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mscaler\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtransform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/frame.py\u001b[0m in \u001b[0;36m__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   3805\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnlevels\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3806\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_getitem_multilevel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3807\u001b[0;31m             \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_loc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3808\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mis_integer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3809\u001b[0m                 \u001b[0mindexer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mindexer\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36mget_loc\u001b[0;34m(self, key, method, tolerance)\u001b[0m\n\u001b[1;32m   3807\u001b[0m                 \u001b[0;31m#  InvalidIndexError. Otherwise we fall through and re-raise\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3808\u001b[0m                 \u001b[0;31m#  the TypeError.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 3809\u001b[0;31m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_check_indexing_error\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   3810\u001b[0m                 \u001b[0;32mraise\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3811\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/core/indexes/base.py\u001b[0m in \u001b[0;36m_check_indexing_error\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m   5923\u001b[0m             \u001b[0;31m# if key is not a scalar, directly raise an error (the code below\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   5924\u001b[0m             \u001b[0;31m# would convert to numpy arrays and raise later any way) - GH29926\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 5925\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mInvalidIndexError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkey\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   5926\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   5927\u001b[0m     \u001b[0;34m@\u001b[0m\u001b[0mcache_readonly\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mInvalidIndexError\u001b[0m: (slice(None, None, None), slice(10, None, None))"
     ]
    }
   ],
   "source": [
    "from sklearn import preprocessing\n",
    "scaler = preprocessing.StandardScaler().fit(X_train[:,10:])\n",
    "x_train[:,10:] = scaler.transform(x_train[:,10:])\n",
    "y_test[:,10:] = scaler.transform(y_test[:,10:])\n",
    "print(x_train.shape)\n",
    "x_train[:10,:]"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
