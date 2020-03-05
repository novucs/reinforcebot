import React from 'react';
import logo from '../logo.png';
import clientBinary from '../clientbinary';
import './App.css';
import Button from "@material-ui/core/Button";

const App = () => (
    <div className="App">
        <img src={logo} alt="logo" className="App-logo"/>
        <Button variant="contained" color="primary" href={clientBinary} download>
            Download Client
        </Button>
    </div>
);

export default App;
