import React from 'react';
import Button from "@material-ui/core/Button";
import TextField from "@material-ui/core/TextField";
import Card from "@material-ui/core/Card";
import {Container} from "@material-ui/core";
import logo from "../logo.png";
import CardActions from "@material-ui/core/CardActions";
import Grid from "@material-ui/core/Grid";


export default class Register extends React.Component {
    constructor(props) {
        super(props);
        this.state = {username: '', password: ''}
    }

    register = (event) => {
        fetch('http://localhost:8080/api/auth/users/', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: this.state.username,
                password: this.state.password,
            }),
        }).then(response => {
            return response.json();
        }).then(body => {
            console.log(body);
            if ('username' in body) {
                console.log(body['username']);
                return;
            }

            if ('password' in body) {
                console.log(body['password']);
                return;
            }

            if (200 <= response.status && response.status < 300) {
                console.log("Register success");
            }
        });
    };

    render() {
        return (
            <Container fixed>
                <img src={logo} alt="logo" className="App-logo"/>
                <Card>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                            <TextField id="standard-basic"
                                       label="Username"
                                       onChange={event => this.setState({username: event.target.value})}/>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField id="standard-password-input"
                                       label="Password"
                                       type="password"
                                       autoComplete="current-password"
                                       onChange={event => this.setState({password: event.target.value})}/>
                        </Grid>
                    </Grid>
                    <CardActions>
                        <Button size="small" onClick={this.register}>
                            Register
                        </Button>
                    </CardActions>
                </Card>
            </Container>
        );
    }
}
