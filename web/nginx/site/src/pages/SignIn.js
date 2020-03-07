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
import displayError from '../Util';
import {Copyright} from "../Copywrite";

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
      username: '',
      usernameError: '',
      password: '',
      passwordError: '',
    }
  }

  submit = (event) => {
    fetch('http://localhost:8080/api/auth/jwt/create/', {
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
      if (response.status < 200 || response.status >= 300) {
        response.json().then(body => {
          // this.setState({usernameError: displayError(body['username'] || '')});
          this.setState({passwordError: displayError(body['detail'] || '')});
        });
        return;
      }

      console.log("SignIn success");
      response.json().then(body => {
        // window.localStorage.setItem('jwtAccess', body['access']);
        // window.localStorage.setItem('jwtRefresh', body['refresh']);
      });
    });
  };

  keyPress = (event) => {
    if (event.keyCode === 13 && this.ableToSubmit(event)) {
      this.submit(event);
    }
  };

  ableToSubmit = (event) => {
    return this.state.username !== ''
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
            Sign in
          </Typography>
          <form className={classes.form} onSubmit={this.submit}>
            <Grid container spacing={2}>
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
              Sign In
            </Button>
            <Grid container justify="flex-end">
              <Grid item>
                <Link href="/signup" variant="body2">
                  Don't have an account? Sign up
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
