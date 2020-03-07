import React from 'react';
import Button from "@material-ui/core/Button";
import TextField from "@material-ui/core/TextField";
import {Container} from "@material-ui/core";
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import CssBaseline from "@material-ui/core/CssBaseline";
import Avatar from "@material-ui/core/Avatar";
import Link from "@material-ui/core/Link";
import Box from "@material-ui/core/Box";
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import withStyles from "@material-ui/core/styles/withStyles";
import {Copyright} from "../Copywrite";
import displayError from '../Util';

// import logo from '../logo.png';

const styles = theme => ({
  paper: {
    marginTop: theme.spacing(8),
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  avatar: {
    margin: theme.spacing(1),
    backgroundColor: theme.palette.secondary.main,
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(3),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
  },
});


class SignUp extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      firstName: '',
      firstNameError: '',
      lastName: '',
      lastNameError: '',
      username: '',
      usernameError: '',
      email: '',
      emailError: '',
      password: '',
      passwordError: '',
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
          this.setState({firstNameError: displayError(body['first_name'] || '')});
          this.setState({lastNameError: displayError(body['last_name'] || '')});
          this.setState({usernameError: displayError(body['username'] || '')});
          this.setState({emailError: displayError(body['email'] || '')});
          this.setState({passwordError: displayError(body['password'] || '')});
        });
        return;
      }

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
    const {classes} = this.props;
    return (
      <Container component="main" maxWidth="xs">
        {/*<img src={logo} alt="logo" className="App-logo"/>*/}
        <CssBaseline/>
        <div className={classes.paper}>
          <Avatar className={classes.avatar}>
            <LockOutlinedIcon/>
          </Avatar>
          <Typography component="h1" variant="h5">
            Sign up
          </Typography>
          <form className={classes.form} onSubmit={this.submit} noValidate>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  id="firstName"
                  name="firstName"
                  label="First Name"
                  variant="outlined"
                  autoComplete="fname"
                  onKeyDown={this.keyPress}
                  error={this.state.firstNameError !== ''}
                  helperText={this.state.firstNameError}
                  onChange={event => this.setState({firstName: event.target.value})}
                  fullWidth
                  required
                  autoFocus
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  id="lastName"
                  name="lastName"
                  label="Last Name"
                  variant="outlined"
                  autoComplete="lname"
                  fullWidth
                  required
                  onKeyDown={this.keyPress}
                  error={this.state.lastNameError !== ''}
                  helperText={this.state.lastNameError}
                  onChange={event => this.setState({lastName: event.target.value})}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  id="userName"
                  name="userName"
                  label="Username"
                  variant="outlined"
                  autoComplete="uname"
                  fullWidth
                  required
                  onKeyDown={this.keyPress}
                  error={this.state.usernameError !== ''}
                  helperText={this.state.usernameError}
                  onChange={event => this.setState({username: event.target.value})}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  id="email"
                  name="email"
                  label="Email Address"
                  variant="outlined"
                  autoComplete="email"
                  fullWidth
                  required
                  onKeyDown={this.keyPress}
                  error={this.state.emailError !== ''}
                  helperText={this.state.emailError}
                  onChange={event => this.setState({email: event.target.value})}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  id="password"
                  name="password"
                  label="Password"
                  type="password"
                  variant="outlined"
                  autoComplete="current-password"
                  fullWidth
                  required
                  onKeyDown={this.keyPress}
                  error={this.state.passwordError !== ''}
                  helperText={this.state.passwordError}
                  onChange={event => this.setState({password: event.target.value})}
                />
              </Grid>
            </Grid>
            <Button
              fullWidth
              variant="contained"
              color="primary"
              disabled={!this.ableToSubmit()}
              className={classes.submit}
              onClick={this.submit}
            >
              Sign Up
            </Button>
            <Grid container justify="flex-end">
              <Grid item>
                <Link href="/signin" variant="body2">
                  Already have an account? Sign in
                </Link>
              </Grid>
            </Grid>
          </form>
        </div>
        <Box mt={5}>
          <Copyright/>
        </Box>
      </Container>
    );
  }
}

export default withStyles(styles)(SignUp);
