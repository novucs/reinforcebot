import React from 'react';
import {BASE_URL, displayErrors, ensureSignedOut, fetchMe, signIn} from '../Util';
import {Button, Form, Grid, Header, Message, Segment} from "semantic-ui-react";
import logo from "../icon.svg";
import TopMenu from "../components/TopMenu";
import Footer from "../components/Footer";

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
    fetch(BASE_URL + '/api/auth/users/', {
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
      if (response.status !== 201) {
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
      signIn(this.state.username, this.state.password, (errors) => {
        this.setState({errors: errors});
      });
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

  componentDidMount = () => {
    ensureSignedOut();
    fetchMe(me => this.setState({me}));
  };

  render = () => (
    <div className='SitePage'>
      <TopMenu me={this.state.me}/>
      <Grid textAlign='center' style={{marginTop: '32px', marginBottom: '32px'}} className='SiteContents'
            verticalAlign='middle'>
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
                Sign up
              </Button>
            </Segment>
          </Form>
          <Message
            error
            header='Sign Up Unsuccessful'
            list={this.state.errors}
            hidden={this.state.errors.length === 0}
          />
          <Message info>
            Already have an account? <a href="/signin">Sign In</a>
          </Message>
        </Grid.Column>
      </Grid>
      <Footer/>
    </div>
  );
}
