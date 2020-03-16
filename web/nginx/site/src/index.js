import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import * as serviceWorker from './serviceWorker';
import {BrowserRouter} from "react-router-dom";
import App from "./components/App";
import 'semantic-ui-css/semantic.css';
import 'react-semantic-toasts/styles/react-semantic-alert.css';
import {loadStripe} from "@stripe/stripe-js";
import {Elements} from "@stripe/react-stripe-js";

const stripePromise = loadStripe('pk_test_oWoryDiggnHg7aCFs1Czg8SI00RQSNsHVN');

ReactDOM.render(
  <BrowserRouter>
    <Elements stripe={stripePromise}>
      <App/>
    </Elements>
  </BrowserRouter>,
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
