import numpy as np
import random as rand
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# helper function to reduce the verbosity of the code to follow
def urv(a=1.0, b=-1.0): return rand.uniform(a,b)

# helper function for adjusting the "learning" rates
def rate_limiter(x): return 1 - np.exp(np.log(0.5) - 5e10*x)


# iterative steepest descent
# b and w have separate learning rates, which are adaptive
# initial learning rates are highly sensitive to the number of data points
def grad(X, Y, w_lr=1e-3, b_lr=5e-2, tol=1e-5):
    num_iter = 0
    max_iter = 1000    

    # choose "random" initial values for b and w
    b_path = [urv(1, 99)]
    w_path = [urv(-9, 9)]
    ones = np.ones(len(X))

    for n in range(max_iter):
        num_iter = n

        w, b = w_path[-1], b_path[-1]
        w_err = X.dot(w*X + b - Y)
        b_err = ones.dot(w*X + b - Y)

        if abs(w_err) < tol:
            w_path.append(w)
        else:
            w_rate = w_lr*rate_limiter(abs(w_err))
            w_path.append(w - w_rate*w_err)

        if abs(b_err) < tol:
            b_path.append(b)
        else:
            b_rate = b_lr*rate_limiter(abs(b_err))
            b_path.append(b - b_rate*b_err)

        if (abs(w_err) < tol) & (abs(b_err) < tol):
            break

    return b_path, w_path, num_iter
# end


class Line2D:
    """ A linear function with optional randomized slope and intercept \\
        f: R -> R ; f(x) = m*x + b"""
    def __init__(self, m=urv(), b=urv()):
        self.m = m
        self.b = b

    def __call__(self, x):
        return x*self.m + self.b
# end


class Model:
    def __init__(self, slope=2, intercept=50):
        self.slope = slope
        self.intercept = intercept
        self.f = Line2D(self.slope, self.intercept)

        # training data for the model
        self.trnX = np.linspace(-10, 10, 10)
        self.trnY = [self.f(x) + urv(-5, 5) for x in self.trnX]

        # testing data for later
        self.tstX = np.linspace(10, 20, 10)
        self.tstY = [self.f(x) + urv(-5, 5) for x in self.tstX]

        # bias and weight vectors
        self.bb = np.arange(0,101,1)
        self.ww = np.arange(-10,11,1)

        # initialize a matrix which will store the model error
        self.Z = np.zeros((len(self.bb), len(self.ww)))

        # error as a function of bias and weight
        for i, b in enumerate(self.bb):
            for j, w in enumerate(self.ww):
                for n, (x, y) in enumerate(zip(self.trnX, self.trnY)):
                    self.Z[(i,j)] += (w*x + b - y)**2
                # end
            # end
        # end

        # "normalize" Z by scaling its components by
        #  the maximum value found in Z
        self.Z = self.Z/self.Z.max()
        self.b_path, self.w_path, self.num_iter = grad(self.trnX, self.trnY)

        self.b = self.b_path[-1]
        self.w = self.w_path[-1]


    def reg(self, x):
        return self.w * x + self.b
    # end


    def loss_plot(self):
        loss_trace = go.Contour(
            x=self.ww, y=self.bb, z=self.Z,
            contours=dict(
                showlabels=True,
                labelfont=dict(
                    size=12,
                    color="gray"
                ),
            ),
            colorscale=[
                [0.0, "mediumturquoise"],
                [0.5, "gold"],
                [1.0, "lightsalmon"]
            ],
            showscale=False,
            line_width=0
        )

        path_trace = go.Scatter(
            x=self.w_path[0:-2], y=self.b_path[0:-2],
            name="gradient descent path",
            mode="lines+markers",
            marker_size=12,
            marker_color="white",
            line_width=5,
            line_color="lightgray",
        )

        true_min_trace = go.Scatter(
            x=[self.slope], y=[self.intercept],
            name="true minimum before noise (w=2, b=50)",
            mode="markers",
            marker_size=15,
            marker_color="lightsalmon",
            marker_symbol="x"
        )

        final_w = round(self.w, 4)
        final_b = round(self.b, 4)

        final_params_trace = go.Scatter(
            x=[self.w_path[-1]], y=[self.b_path[-1]],
            name=f"final parameters (w={final_w}, b={final_b})",
            mode="markers",
            marker_size=15,
            marker_color="gold",
            marker_symbol="x"
        )

        iters = f"n={self.num_iter} iterations</b>"
        ttl = f"<b>Gradient descent path on error surface, {iters}"
        fig = go.Figure(
            data=[loss_trace, path_trace, true_min_trace, final_params_trace],
            layout=go.Layout(
                width=950, height=950,
                title=ttl,
                xaxis_title="weight values",
                yaxis_title="bias values",
                legend_x=0, legend_y=1,
                legend_bgcolor="rgba(0,0,0,0.3)",
                legend_font_color="white"
            )
        )

        fig.show()


    def error_plot(self):
        target = self.f(self.tstX)
        predict = self.reg(self.tstX)

        predicted_trace = go.Scatter(
            x=self.tstX, y=predict,
            line=dict(width=5),
            name="predicted",
            mode="markers",
            marker_size=10,
            marker_color="mediumturquoise",
            marker_symbol="x"
        )

        target_trace = go.Scatter(
            x=self.tstX, y=target,
            name="target",
            mode="markers",
            marker_size=15,
            marker_color="gold"
        )

        error_trace = go.Scatter(
            x=self.tstX, y=abs(target-predict),
            name="error",
            mode="lines",
            line_width=5,
            line_color="lightsalmon"
        )

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.8, 0.2],
            subplot_titles=("<b>Predicted values vs real values</b>", ""),
            vertical_spacing=0.0075, shared_xaxes=True
        )

        fig.add_trace(target_trace, row=1, col=1)
        fig.add_trace(predicted_trace, row=1, col=1)
        fig.add_trace(error_trace, row=2, col=1)

        fig.update_xaxes(
            row=1, col=1,
            gridcolor="rgb(220,220,220)",
            zeroline=False
        )

        fig.update_xaxes(
            title_text="<b>X</b>",
            row=2, col=1,
            gridcolor="rgb(220,220,220)",
            zeroline=False
        )

        fig.update_yaxes(
            title_text="<b>Y</b>",
            row=1, col=1,
            gridcolor="rgb(220,220,220)"
        )

        fig.update_yaxes(
            title_text="<b>ABSOLUTE ERROR</b>",
            titlefont_color="lightsalmon",
            row=2, col=1,
            gridcolor="rgb(220,220,220)",
            zeroline=False
        )

        fig.update_layout(
            height=950,
            width=950,
            plot_bgcolor="rgb(230,230,230)"
        )

        fig.show()