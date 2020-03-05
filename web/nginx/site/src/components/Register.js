import React from 'react';
import clientBinary from '../clientbinary';
import Button from "@material-ui/core/Button";

const Register = () => (
    <div>
        <Button variant="contained" color="primary" href={clientBinary} download>
            Register
        </Button>
    </div>
);

export default Register;
