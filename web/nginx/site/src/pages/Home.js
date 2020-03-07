import React from 'react';
import logo from '../logo.png';
import clientBinary from '../clientbinary';
import '../components/App.css';
import Button from "@material-ui/core/Button";
import {Link} from "react-router-dom";

const Home = () => (
    <div className="App">
        <img src={logo} alt="logo" className="App-logo"/>
        <Button variant="contained" color="primary" href={clientBinary} download>
            Download Client
        </Button>
        <Link to="/signup">
            <Button variant="contained" color="primary">
                Sign up
            </Button>
        </Link>
        <Link to="/signin">
            <Button variant="contained" color="primary">
                Sign in
            </Button>
        </Link>
    </div>
);

export default Home;
