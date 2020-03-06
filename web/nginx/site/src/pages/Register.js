import React from 'react';
import clientBinary from '../clientbinary';
import Button from "@material-ui/core/Button";
import TextField from "@material-ui/core/TextField";
import Card from "@material-ui/core/Card";
import {Container} from "@material-ui/core";
import logo from "../logo.png";
import CardActions from "@material-ui/core/CardActions";
import Typography from "@material-ui/core/Typography";
import {makeStyles} from '@material-ui/core/styles';
import Grid from "@material-ui/core/Grid";

const useStyles = makeStyles({});

export default function Register() {
    const classes = useStyles();
    return (
        <Container fixed>
            <img src={logo} alt="logo" className="App-logo"/>
            <Card>
                <form noValidate autoComplete="off">
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <TextField id="standard-basic"
                                       label="Username"/>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField id="standard-password-input"
                                       label="Password"
                                       type="password"
                                       autoComplete="current-password"/>
                        </Grid>
                    </Grid>
                </form>
                <CardActions>
                    <Button size="small" href={clientBinary} download>
                        Register
                    </Button>
                </CardActions>
            </Card>
        </Container>
    );
};
