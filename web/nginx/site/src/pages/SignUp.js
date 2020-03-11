import React from 'react';
// import Button from "@material-ui/core/Button";
// import TextField from "@material-ui/core/TextField";
// import {Container} from "@material-ui/core";
// import Grid from "@material-ui/core/Grid";
// import Typography from "@material-ui/core/Typography";
// import CssBaseline from "@material-ui/core/CssBaseline";
// import Avatar from "@material-ui/core/Avatar";
// import Link from "@material-ui/core/Link";
// import Box from "@material-ui/core/Box";
// import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
// import withStyles from "@material-ui/core/styles/withStyles";
// import {Copyright} from "../Copywrite";
import displayError, {displayErrors} from '../Util';
import {Button, Form, Grid, Header, Message, Segment} from "semantic-ui-react";
import logo from "../icon.svg";

// import logo from '../logo.png';

// const styles = theme => ({
//   paper: {
//     marginTop: theme.spacing(8),
//     display: 'flex',
//     flexDirection: 'column',
//     alignItems: 'center',
//   },
//   avatar: {
//     margin: theme.spacing(1),
//     backgroundColor: theme.palette.secondary.main,
//   },
//   form: {
//     width: '100%', // Fix IE 11 issue.
//     marginTop: theme.spacing(3),
//   },
//   submit: {
//     margin: theme.spacing(3, 0, 2),
//   },
// });


export default class SignUp extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      firstName: '',
      lastName: '',
      username: '',
      email: '',
      password: '',
      errors: [],
    }
  }

  submit = (event) => {
    fetch('http://localhost:8080/api/auth/users/', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        first_name: this.state.firstName,
        last_name: this.state.lastName,
        username: this.state.username,
        email: this.state.email,
        password: this.state.password,
      }),
    }).then(response => {
      if (response.status < 200 || response.status >= 300) {
        response.json().then(body => {
          this.setState({
            errors: displayErrors(
              body['first_name'],
              body['last_name'],
              body['username'],
              body['email'],
              body['password'])
          });
        });
        return;
      }

      this.setState({errors: []});
      console.log("SignUp success");
    });
  };

  keyPress = (event) => {
    if (event.keyCode === 13 && this.ableToSubmit(event)) {
      this.submit(event);
    }
  };

  ableToSubmit = (event) => {
    return this.state.firstName !== ''
      && this.state.lastName !== ''
      && this.state.username !== ''
      && this.state.email !== ''
      && this.state.password !== '';
  };

  render() {
    return (
      <Grid textAlign='center' style={{height: '100vh'}} verticalAlign='middle'>
        <Grid.Column style={{maxWidth: 450}}>
          <Header as="h2" color="teal" textAlign="center">
            <img src={logo} alt="logo" className="image"/>{" "}
            Create a new account
          </Header>
          <Form size="large">
            <Segment stacked>
              <Form.Input
                fluid
                placeholder="First name"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({firstName: event.target.value})}
              />
              <Form.Input
                fluid
                placeholder="Last name"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({lastName: event.target.value})}
              />
              <Form.Input
                fluid
                icon="envelope"
                iconPosition="left"
                placeholder="Email"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({email: event.target.value})}
              />
              <Form.Input
                fluid
                icon="user"
                iconPosition="left"
                placeholder="Username"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({username: event.target.value})}
              />
              <Form.Input
                fluid
                icon="lock"
                iconPosition="left"
                placeholder="Password"
                type="password"
                onKeyDown={this.keyPress}
                onChange={event => this.setState({password: event.target.value})}
              />
              <Button
                color="teal"
                fluid size="large"
                disabled={!this.ableToSubmit()}
                onClick={this.submit}
              >
                Sign in
              </Button>
            </Segment>
          </Form>
          <Message
            error
            header='Sign Up Unsuccessful'
            list={this.state.errors}
            hidden={this.state.errors.length === 0}
          />
          <Message>
            New to us? <a href="/signin">Sign In</a>
          </Message>
        </Grid.Column>
      </Grid>
    );
  }
}
